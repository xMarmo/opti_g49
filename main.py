import pandas as pd
import os
from gurobipy import GRB, Model, quicksum

# 1. CARGA DE DATOS

print("--- INICIANDO MODELO DE OPTIMIZACIN ---")
print("Cargando parametros desde carpeta 'data'...")

DATA_DIR = 'data'

def cargar_parametro(archivo, cols_indice, col_valor):
    ruta = os.path.join(DATA_DIR, archivo)
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"Falta el archivo: {ruta}")
    
    try:
        df = pd.read_csv(ruta, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(ruta, encoding='latin-1')
    
    df.columns = [c.strip() for c in df.columns]
    
    if isinstance(cols_indice, list):
        return df.set_index(cols_indice)[col_valor].to_dict()
    else:
        return df.set_index(cols_indice)[col_valor].to_dict()

try:
    
    # c_it: Costo monitoreo (t, i, Cit)
    df_cit = pd.read_csv(os.path.join(DATA_DIR, 'c_it.csv'))
    C = df_cit.set_index(['i', 't'])['Cit'].to_dict()

    # f_i: Frecuencia requerida (i, f_i)
    f = cargar_parametro('f_i.csv', 'i', 'f_i')

    # F_ot: Disponibilidad Operador (t, o, Fot)
    df_fot = pd.read_csv(os.path.join(DATA_DIR, 'f_ot.csv'))
    F = df_fot.set_index(['o', 't'])['Fot'].to_dict()

    # g_dt: Costo Mantencion Dron (t, d, gdt)
    df_gdt = pd.read_csv(os.path.join(DATA_DIR, 'g_dt.csv'))
    g = df_gdt.set_index(['d', 't'])['gdt'].to_dict()

    # h_t: Costo Operador (t, Día, ht)
    df_ht = pd.read_csv(os.path.join(DATA_DIR, 'h_t.csv'), usecols=['t', 'ht'])
    h = df_ht.set_index('t')['ht'].to_dict()

    # K_t: Capacidad Mantencion (t, Kt)
    K = cargar_parametro('K_t.csv', 't', 'Kt')

    # P_i: Penalizacion (i, Pi)
    P = cargar_parametro('P_i.csv', 'i', 'Pi')

    # Q_i: Tiempo minimo (i, Qi)
    Q = cargar_parametro('Q_i.csv', 'i', 'Qi')

    # R_i: Riesgo (i, Ri)
    R = cargar_parametro('R_i.csv', 'i', 'Ri')

except Exception as e:
    print(f"\n[ERROR CRITICO] Fallo la carga de datos: {e}")
    exit()

# --- Conjuntos Definidos y Escalares ---
B = 128_000_000   # Presupuesto 
a = 0.67          # Autonomia vuelo (hrs)
U = 300.0         # Vida util (hrs)
H = 1             # Drones por operador

# Rangos
I = range(1, 6)    # 5 Relaves
D = range(1, 16)   # 15 Drones
T = range(1, 366)  # 365 Días
O = range(1, 4)    # 3 Operadores

# Big-M
BigM_horas = 2 * U
BigM_drones = len(D)

# Estado inicial (asumimos drones nuevos con 0 uso)
E0 = {d: 0.0 for d in D}

print("Datos cargados exitosamente. Construyendo modelo...")

# 2. MODELO GUROBI

model = Model("Optimizacion_Monitoreo_Relaves")
model.setParam('TimeLimit', 1800) # 30 min limite

# --- Variables ---
x = model.addVars(I, T, vtype=GRB.BINARY, name="x")
z = model.addVars(I, D, T, vtype=GRB.BINARY, name="z")
q = model.addVars(I, D, T, vtype=GRB.CONTINUOUS, name="q")
m = model.addVars(D, T, vtype=GRB.BINARY, name="m")
u_op = model.addVars(T, vtype=GRB.INTEGER, name="u")
E = model.addVars(D, T, vtype=GRB.CONTINUOUS, name="E")
l = model.addVars(I, vtype=GRB.CONTINUOUS, name="l")

# --- Función Objetivo ---
obj_beneficio = quicksum(R[i] * x[i,t] for i in I for t in T)
obj_penalizacion = quicksum(P[i] * l[i] for i in I)

model.setObjective(obj_beneficio - obj_penalizacion, GRB.MAXIMIZE)

# --- Restricciones ---

# R1: Presupuesto
costo_mon = quicksum(C.get((i,t), 0) * x[i,t] for i in I for t in T)
costo_man = quicksum(g.get((d,t), 0) * m[d,t] for d in D for t in T)
costo_ops = quicksum(h.get(t, 0) * u_op[t] for t in T)

model.addConstr(costo_mon + costo_man + costo_ops <= B, name="R1_Presupuesto")

# R2: Capacidad Supervision
model.addConstrs((quicksum(z[i,d,t] for i in I for d in D) <= H * u_op[t] for t in T), name="R2_Sup")

# R3: Disponibilidad Operadores
model.addConstrs((u_op[t] <= quicksum(F.get((o,t), 0) for o in O) for t in T), name="R3_DispOp")

# R4: Autonomia Diaria
model.addConstrs((quicksum(q[i,d,t] for i in I) <= a * (1 - m[d,t]) for d in D for t in T), name="R4_Autonomia")

# Acumulacion Desgaste (E)
for d in D:
    # Caso t=1 (Usamos E0)
    vuelo_t1 = quicksum(q[i,d,1] for i in I)
    model.addConstr(E0[d] + vuelo_t1 <= U + BigM_horas * m[d,1], name=f"R5_ini_d{d}")
    model.addConstr(E[d,1] <= U * (1 - m[d,1]), name=f"R6_ini_d{d}")
    model.addConstr(E[d,1] >= E0[d] + vuelo_t1 - BigM_horas * m[d,1], name=f"R7_ini_d{d}")
    model.addConstr(E[d,1] <= E0[d] + vuelo_t1 + BigM_horas * m[d,1], name=f"R8_ini_d{d}")

# Caso t > 1
periods_post = [t for t in T if t > 1]
model.addConstrs((E[d,t-1] + quicksum(q[i,d,t] for i in I) <= U + BigM_horas * m[d,t] for d in D for t in periods_post), name="R5_Mant")
model.addConstrs((E[d,t] <= U * (1 - m[d,t]) for d in D for t in periods_post), name="R6_Reset")
model.addConstrs((E[d,t] >= E[d,t-1] + quicksum(q[i,d,t] for i in I) - BigM_horas * m[d,t] for d in D for t in periods_post), name="R7_Low")
model.addConstrs((E[d,t] <= E[d,t-1] + quicksum(q[i,d,t] for i in I) + BigM_horas * m[d,t] for d in D for t in periods_post), name="R8_Up")

# R9: Capacidad Operativa Drones
model.addConstrs((quicksum(z[i,d,t] for i in I) <= 1 - m[d,t] for d in D for t in T), name="R9_Cap_Dron")

# R10: Asignacion
model.addConstrs((quicksum(z[i,d,t] for d in D) >= x[i,t] for i in I for t in T), name="R10_Min")
model.addConstrs((quicksum(z[i,d,t] for d in D) <= BigM_drones * x[i,t] for i in I for t in T), name="R10_Max")

# R12: Tiempo Minimo
model.addConstrs((quicksum(q[i,d,t] for d in D) >= Q[i] * x[i,t] for i in I for t in T), name="R12_Qmin")

# R13: Faltantes
model.addConstrs((l[i] >= f[i] - quicksum(x[i,t] for t in T) for i in I), name="R13_Faltantes")
model.addConstrs((l[i] >= 0 for i in I), name="R13_Pos")

# R14: Capacidad Mantenimiento Global
model.addConstrs((quicksum(m[d,t] for d in D) <= K[t] for t in T), name="R14_K")

# R15: Espaciamiento Dinamico (Gap)
print("Generando restricciones de espaciamiento...")
for i in I:
    freq = f[i]
    if freq > 1:
        gap_i = int((365 / freq) * 0.8)
        gap_i = max(1, gap_i) # Minimo 1 dia
        
        # Ventana movil
        for t_start in range(1, 366 - gap_i + 2):
            window = range(t_start, t_start + gap_i)
            # Solo sumar dias que estén dentro del horizonte T
            vars_in_window = [x[i,k] for k in window if k in T]
            if vars_in_window:
                model.addConstr(quicksum(vars_in_window) <= 1, name=f"R15_Gap_i{i}_t{t_start}")

# 3. EJECUCION
print("\nIniciando Optimización en Gurobi...")
model.optimize()

if model.Status == GRB.OPTIMAL:
    print("\n" + "="*50)
    print(f"  RESULTADO ÓPTIMO: {model.objVal:,.2f}")
    print("="*50)
    
    total_cost = costo_mon.getValue() + costo_man.getValue() + costo_ops.getValue()
    print(f"Presupuesto Utilizado: ${total_cost:,.0f} / ${B:,.0f} ({total_cost/B*100:.1f}%)")
    
    print("\n--- Resumen por Relave ---")
    for i in I:
        n_visitas = sum(x[i,t].X for t in T)
        estado = "CUMPLIDO" if l[i].X < 0.1 else f"FALTAN {l[i].X:.0f}"
        print(f"Relave {i} (Riesgo {R[i]}, Meta {f[i]}): {n_visitas:.0f} visitas -> {estado}")

    print("\n--- Uso de Drones ---")
    drones_mant = sum(m[d,t].X for d in D for t in T)
    print(f"Días totales en mantenimiento (toda la flota): {drones_mant:.0f}")

else:
    print("\nNo se encontró solución óptima. Estado:", model.Status)
    if model.Status == GRB.INFEASIBLE:
        model.computeIIS()
        model.write("error_modelo.ilp")
        print("Revisa 'error_modelo.ilp' para ver las restricciones conflictivas.")