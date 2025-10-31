from gurobipy import GRB, Model
from gurobipy import quicksum

#inicialización modelo
model = Model("Optimizacion_Monitoreo_Relaves")
model.setParam('TimeLimit', 300)

# Conjuntos
I = range(1, 11) # relaves
T = range(1, 366) # dias
D = range(1, 6) # drones
O = range(1, 4) # operadores

# conjunto de tiempo extendido para manejar variable E_dt
T_con_0 = range(0, 366)


# Parametros
B = 1000000  # presupuesto
alpha = 0.01 # ponderador de costos
H = 2 # max drones por operador
U = 100 # horas max de uso antes de mantencion
R = {i: 10 for i in I} # riesgo del relave i
f = {i: 5 for i in I} # frecuencia minima de monitoreo
P = {i: 50 for i in I} # penalizacion monitoreos faltantes
Q = {i: 0.5 for i in I} # tiempo minimo de monitoreo (hrs)
n = {i: 1 if i % 2 == 0 else 0 for i in I} # 1 si relave i es de alto riesgo
a = {d: 4 for d in D} # autonomia de vuelo (hrs)
E0 = {d: 0 for d in D} # horas de uso acumulado inicial (en t=0)
K = {t: 1 for t in T} # max drones en mantenimiento por dia 
h = {t: 100 for t in T} # costo por operador por dia
C = {(i, t): 50 for i in I for t in T} # costo monitoreo i en t
F = {(o, t): 1 for o in O for t in T} # disponibilidad operador o en t
g = {(d, t): 75 for d in D for t in T} # costo mantenimiento d en t


# Variables
# x_it: 1 si se monitorea relave i en periodo t
x = model.addVars(I, T, vtype=GRB.BINARY, name="x_monitoreo")

# z_idt: 1 si el dron d visita relave i en periodo t
z = model.addVars(I, D, T, vtype=GRB.BINARY, name="z_dron_visita_relave")

# q_idt: horas de monitoreo de dron d a relave i en t
q = model.addVars(I, D, T, vtype=GRB.CONTINUOUS, name="q_horas_monitoreo")

# m_dt : 1 si dron d esta en mantenimiento en t
m = model.addVars(D, T, vtype=GRB.BINARY, name="m_mantenimiento")

# u_t: numero de operadores usados en t
u = model.addVars(T, vtype=GRB.INTEGER, name="u_n_operadores")

# E_dt: uso acumulado de dron d al final de t
E = model.addVars(D, T_con_0, vtype=GRB.CONTINUOUS, name="E_uso_acumulado")

# l_i: numero de monitoreos faltantes en relave i
l = model.addVars(I, vtype=GRB.CONTINUOUS, name="l_monitoreos_faltantes")

# Funcion objetivo

# beneficio por riesgo de monitoreo
termino_beneficio = quicksum(R[i] * x[i,t] for i in I for t in T)

# costos operativos (ponderados por alpha)
costo_monitoreo = quicksum(C[i,t] * x[i,t] for i in I for t in T)
costo_mantenimiento = quicksum(g[d,t] * m[d,t] for d in D for t in T)
costo_operadores = quicksum(h[t] * u[t] for t in T)

termino_costos = alpha * (costo_monitoreo + costo_mantenimiento + costo_operadores)

# penalizacion por monitoreos faltantes
termino_penalizacion = quicksum(P[i] * l[i] for i in I)

# Objetivo: maximizar el beneficio neto
model.setObjective(
    termino_beneficio - termino_costos - termino_penalizacion,
    GRB.MAXIMIZE
)


# Restricciones

print("Agregando restricciones...")

# R1: presupuesto
model.addConstr(
    costo_monitoreo + costo_mantenimiento + costo_operadores <= B,
    name="presupuesto"
)

# R2: capacidad de supervision
model.addConstrs(
    (quicksum(z[i,d,t] for i in I for d in D) <= H * u[t] for t in T),
    name="capacidad_supervision"
)

# R3: limite de disponibilidad de operadores 
model.addConstrs(
    (u[t] <= quicksum(F[o,t] for o in O) for t in T),
    name="disponibilidad_operadores"
)

# R4: autonomia por vuelo
model.addConstrs(
    (q[i,d,t] <= a[d] * z[i,d,t] for i in I for d in D for t in T),
    name="autonomia_vuelo"
)

# R5: autonomia por dron (dia) y efecto mantenimiento
model.addConstrs(
    (quicksum(q[i,d,t] for i in I) <= a[d] * (1 - m[d,t]) for d in D for t in T),
    name="autonomia_diaria_dron"
)

# R6: uso acumulado del dron
# Valor inicial t=0
model.addConstrs(
    (E[d,0] == E0[d] for d in D),
    name="uso_acumulado_inicial"
)
# Valor para t>=1
model.addConstrs(
    (E[d,t] == (E[d,t-1] + quicksum(q[i,d,t] for i in I)) * (1 - m[d,t]) for d in D for t in T),
    name="uso_acumulado_dron"
)

# R7: limite de uso (antes de mantenimiento)
model.addConstrs(
    (E[d,t] <= U for d in D for t in T),
    name="limite_uso"
)

# R8: capacidad total de drones operativos por periodo
model.addConstrs(
    (quicksum(z[i,d,t] for i in I for d in D) <= quicksum(1 - m[d,t] for d in D) for t in T),
    name="drones_operativos"
)

# R9: asignacion dron - relave
model.addConstrs(
    (x[i,t] <= quicksum(z[i,d,t] for d in D) for i in I for t in T),
    name="asignacion_dron_relave"
)

# R10: cada dron realiza a lo sumo una mision simultanea
model.addConstrs(
    (quicksum(z[i,d,t] for i in I) <= 1 for d in D for t in T),
    name="mision_unica_dron"
)

# R11: minimo tiempo de visita necesario
model.addConstrs(
    (quicksum(q[i,d,t] for d in D) >= Q[i] * x[i,t] for i in I for t in T),
    name="minimo_tiempo_visita"
)

# R12: monitoreos faltantes
model.addConstrs(
    (l[i] >= f[i] * n[i] - quicksum(x[i,t] for t in T) for i in I),
    name="monitoreos_faltantes"
)
# Monitoreos faltantes no negativos
model.addConstrs(
    (l[i] >= 0 for i in I),
    name="no_negatividad_faltantes"
)

# R13: maxima cantidad de drones en mantenimiento
model.addConstrs(
    (quicksum(m[d,t] for d in D) <= K[t] for t in T),
    name="max_mantenimiento"
)


# Optimizacion
print("Restricciones agregadas, optimizando...")
model.optimize()

# Resultados
if model.Status == GRB.OPTIMAL:
    print("\nSolucion optima encontrada.")
    print(f"Valor objetivo: {model.objVal:,.2f}")

    print("\n--- Monitoreos a realizar (x_it) ---")
    for t in T:
        for i in I:
            if x[i,t].X > 0.5:
                print(f"Dia {t}: Monitorear relave {i}")
    
    print("\n--- Operadores a usar (u_t) ---")
    for t in T:
        if u[t].X > 0.1:
            print(f"Dia {t}: Usar {u[t].X:.0f} operadores")

    print("\n--- Monitores faltantes (l_i) ---")
    for i in I:
        if l[i].X > 0.1:
            print(f"Relave {i}: Faltaron {l[i].X:.1f} monitoreos")

elif model.Status == GRB.INFEASIBLE:
    print("\nEl modelo es infactible.")

elif model.Status == GRB.TIME_LIMIT:
    print("\nSe alcanzo el limite de tiempo sin encontrar solucion optima.")

else: 
    print(f"\nOptimizacion finalizada con estado: {model.Status}")