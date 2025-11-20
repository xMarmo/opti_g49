from gurobipy import GRB, Model, quicksum
import math

# FALTA IMPORTAR LOS DATOS 

from datos import * # Inicialización del modelo
model = Model("Optimizacion_Monitoreo_Relaves")
model.setParam('TimeLimit', 1800)

# Constante Big-M
BigM = 10000 
M_drones = len(D) # M para la restricción de asignación (cantidad de drones)


# 1. VARIABLES DE DECISIÓN

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

# E_dt: uso acumulado de dron d al final de t (Requiere T_con_0 para condición inicial)
# Asumimos que T_con_0 incluye el 0 y todos los t de T
E = model.addVars(D, T_con_0, vtype=GRB.CONTINUOUS, name="E_uso_acumulado")

# l_i: numero de monitoreos faltantes en relave i
l = model.addVars(I, vtype=GRB.CONTINUOUS, name="l_monitoreos_faltantes")


# 2. FUNCIÓN OBJETIVO

# Beneficio por riesgo de monitoreo
termino_beneficio = quicksum(R[i] * x[i,t] for i in I for t in T)

# Costos operativos (ponderados por alpha)
costo_monitoreo = quicksum(C[i,t] * x[i,t] for i in I for t in T)
costo_mantenimiento = quicksum(g[d,t] * m[d,t] for d in D for t in T)
costo_operadores = quicksum(h[t] * u[t] for t in T)

termino_costos = alpha * (costo_monitoreo + costo_mantenimiento + costo_operadores)

# Penalización por monitoreos faltantes
termino_penalizacion = quicksum(P[i] * l[i] for i in I)

# Objetivo: Maximizar Beneficio Neto
model.setObjective(
    termino_beneficio - termino_costos - termino_penalizacion,
    GRB.MAXIMIZE
)


# 3. RESTRICCIONES
print("Agregando restricciones...")

# --- R1: Presupuesto ---
model.addConstr(
    costo_monitoreo + costo_mantenimiento + costo_operadores <= B,
    name="R1_Presupuesto"
)

# --- R2: Capacidad de supervision ---
model.addConstrs(
    (quicksum(z[i,d,t] for i in I for d in D) <= H * u[t] for t in T),
    name="R2_Capacidad_Supervision"
)

# --- R3: Limite de disponibilidad de operadores ---
model.addConstrs(
    (u[t] <= quicksum(F[o,t] for o in O) for t in T),
    name="R3_Disponibilidad_Operadores"
)

# --- R4: Autonomia por dron (dia) y efecto mantenimiento ---
# La suma de horas de vuelo de un dron no puede superar su autonomía
# Y debe ser 0 si está en mantenimiento
model.addConstrs(
    (quicksum(q[i,d,t] for i in I) <= a[d] * (1 - m[d,t]) for d in D for t in T),
    name="R4_Autonomia_Diaria"
)

# --- BLOQUE DE USO ACUMULADO (R5, R6, R7, R8) ---
# Condicion inicial (t=0)
model.addConstrs(
    (E[d,0] == E0[d] for d in D),
    name="Condicion_Inicial_E"
)

# R5 (Implícito en R7 y R8): Definición de acumulación linealizada
# E_{t-1} + vuelo_actual <= U + M * m_dt
# Si m=0 (operativo), E + q <= U (se cumple el límite)
# Si E + q > U, obliga a m=1.
model.addConstrs(
    (E[d,t-1] + quicksum(q[i,d,t] for i in I) <= U + BigM * m[d,t] for d in D for t in T),
    name="R5_Forzar_Mantenimiento"
)

# R6: Forzar a cero el uso acumulado si hay mantenimiento
model.addConstrs(
    (E[d,t] <= U * (1 - m[d,t]) for d in D for t in T),
    name="R6_Reset_Mantenimiento"
)

# R7: Límite inferior (Acumular horas si NO hay mantencion)
# E_t >= E_{t-1} + q - M*m
model.addConstrs(
    (E[d,t] >= E[d,t-1] + quicksum(q[i,d,t] for i in I) - BigM * m[d,t] for d in D for t in T),
    name="R7_Limite_Inferior_Acumulacion"
)

# R8: Limite superior (Acumular horas si NO hay mantención)
# E_t <= E_{t-1} + q + M*m
model.addConstrs(
    (E[d,t] <= E[d,t-1] + quicksum(q[i,d,t] for i in I) + BigM * m[d,t] for d in D for t in T),
    name="R8_Limite_Superior_Acumulacion"
)

# --- R9: Capacidad total de drones operativos ---
# Si m=1, sum(z) <= 0 (nadie vuela). Si m=0, sum(z) <= 1 (max 1 mision)
model.addConstrs(
    (quicksum(z[i,d,t] for i in I) <= 1 - m[d,t] for d in D for t in T),
    name="R9_Capacidad_Operativa"
)

# --- R10: Asignación Dron - Relave ---
# Parte A: Si x=0, entonces sum(z) <= 0 (nadie va). BigM aqui es numero de drones
model.addConstrs(
    (quicksum(z[i,d,t] for d in D) <= M_drones * x[i,t] for i in I for t in T),
    name="R10_Consistencia_BigM"
)
# Parte B: Si x=1, entonces sum(z) >= 1 (alguien tiene que ir)
model.addConstrs(
    (quicksum(z[i,d,t] for d in D) >= x[i,t] for i in I for t in T),
    name="R10_Asignacion_Minima"
)

# --- R11: Unicidad de Mision ---
model.addConstrs(
    (quicksum(z[i,d,t] for i in I) <= 1 for d in D for t in T),
    name="R11_Mision_Unica"
)

# --- R12: Tiempo total y mínimo de visita ---
model.addConstrs(
    (quicksum(q[i,d,t] for d in D) >= Q[i] * x[i,t] for i in I for t in T),
    name="R12_Tiempo_Minimo"
)

# --- R13: Monitoreos Faltantes ---
model.addConstrs(
    (l[i] >= f[i] - quicksum(x[i,t] for t in T) for i in I),
    name="R13_Calculo_Faltantes"
)
# No negatividad explícita para l[i] (aunque la variable ya es >=0 por defecto en Gurobi)
model.addConstrs(
    (l[i] >= 0 for i in I),
    name="R13_No_Negatividad_l"
)

# --- R14: Máximo mantenimiento simultáneo ---
model.addConstrs(
    (quicksum(m[d,t] for d in D) <= K[t] for t in T),
    name="R14_Capacidad_Mantenimiento"
)

# --- R15: Espaciamiento Temporal (Gap Dinámico) ---
# Calculamos el gap y aplicamos ventana móvil
print("Generando restricciones de espaciamiento...")
for i in I:
    # Solo si se requiere más de 1 visita (para evitar división por cero o lógica innecesaria)
    if f[i] > 1:
        # Cálculo del Gap según fórmula del informe
        gap_i = int((365 / f[i]) * 0.8)
        
        # Seguridad: el gap debe ser al menos 1 día
        if gap_i < 1: gap_i = 1
        
        # Iteramos t cuidando el rango para no salirnos de T (1 a 365)
        # Rango final: 365 - gap + 1. En python range el tope es exclusivo, así que +2
        for t in range(1, 365 - gap_i + 2):
            # Definimos la ventana de días [t, t + gap_i - 1]
            # Convertimos a lista para quicksum
            ventana = range(t, t + gap_i) 
            
            # La suma de visitas en esa ventana debe ser <= 1
            model.addConstr(
                quicksum(x[i, k] for k in ventana if k in T) <= 1,
                name=f"R15_Espaciamiento_Relave{i}_Dia{t}"
            )

# ==========================================
# 4. SOLUCIÓN E IMPRESIÓN
# ==========================================
print("Optimizando...")
model.optimize()

if model.Status == GRB.OPTIMAL:
    print("\n--- SOLUCIÓN ÓPTIMA ENCONTRADA ---")
    print(f"Valor Objetivo: {model.objVal:,.2f}")

    print("\n--- Resumen de Actividad ---")
    for i in I:
        visitas = sum(x[i,t].X for t in T)
        if visitas > 0.5:
            print(f"Relave {i}: {visitas:.0f} visitas realizadas (Meta: {f[i]}, Faltan: {l[i].X:.1f})")
            # Opcional: Imprimir fechas exactas para verificar espaciamiento
            # fechas = [t for t in T if x[i,t].X > 0.5]
            # print(f"   Fechas: {fechas}")

    print("\n--- Uso de Operadores (Muestra) ---")
    dias_con_ops = [t for t in T if u[t].X > 0.1]
    print(f"Días con actividad operativa: {len(dias_con_ops)}")

elif model.Status == GRB.INFEASIBLE:
    print("\nEl modelo es infactible. Revisando IIS...")
    model.computeIIS()
    model.write("modelo_infactible.ilp")
    print("Revisa el archivo 'modelo_infactible.ilp' para ver el conflicto.")
    
else:
    print(f"\nEstado de la optimización: {model.Status}")