import glob
import os
import matplotlib.pyplot as plt
import pandas as pd

# 1. Neueste Logdatei finden
csv_files = glob.glob("smart_cockpit_log_*.csv")

if not csv_files:
  print("Keine Logdatei gefunden!")
  exit()

latest_file = max(csv_files, key=os.path.getctime)
print(f"Lese Logdatei ein: {latest_file}")

# 2. Daten einlesen
df = pd.read_csv(latest_file)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df_valid = df.dropna(subset=["Timestamp", "FPS"]).copy()

if len(df_valid) == 0:
  print("Die Logdatei enthält keine gültigen Daten.")
  exit()

# --- BERECHNUNG DER METRIKEN & SZENARIEN ---
SPEED_MS = 130 * 1000 / 3600  # 130 km/h = ~36.1 m/s
avg_fps = df_valid['FPS'].mean()

# Sekundenschlaf (EAR < 0.1)
microsleep_frames = df_valid[df_valid["EAR"] < 0.1]
total_microsleep_time_sec = len(microsleep_frames) * (1.0 / avg_fps) if avg_fps > 0 else 0
microsleep_meters_at_130 = total_microsleep_time_sec * SPEED_MS

# Handynutzung (Objekterkennung)
phone_frames = df_valid[df_valid["Detected_Objects"].str.contains("cell phone", na=False)]
phone_time_sec = len(phone_frames) * (1.0 / avg_fps) if avg_fps > 0 else 0
phone_meters_at_130 = phone_time_sec * SPEED_MS

# Normaler Zustand
normal_frames = len(df_valid) - len(microsleep_frames) - len(phone_frames)
normal_time_sec = max(0, normal_frames) * (1.0 / avg_fps) if avg_fps > 0 else 0
normal_meters_at_130 = normal_time_sec * SPEED_MS


# --- PLOT 1: Technisches Performance-Dashboard ---
fig1, axes1 = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
fig1.suptitle(
    f"DMS & Object Detection Performance (Hardware: RTX 3050)\nDatei: {latest_file}",
    fontsize=13, fontweight="bold"
)

# EAR & MAR
axes1[0].plot(df_valid["Timestamp"], df_valid["EAR"], label="EAR (Eyes)", color="green", linewidth=1.2)
axes1[0].axhline(0.22, color="orange", linestyle="--", label="Blink-Schwelle (0.22)")
axes1[0].axhline(0.10, color="red", linestyle="--", label="Microsleep (<0.10)")
axes1[0].plot(df_valid["Timestamp"], df_valid["MAR"], label="MAR (Mouth)", color="purple", linewidth=1.2)
axes1[0].set_ylabel("Ratio")
axes1[0].set_title("Driver State Monitoring (EAR & MAR)")
axes1[0].legend(loc="upper right", fontsize=9)
axes1[0].grid(True, alpha=0.3)

# FPS & GPU
ax1_twin = axes1[1].twinx()
line1 = axes1[1].plot(df_valid["Timestamp"], df_valid["FPS"], label="FPS", color="blue", linewidth=1.2)
line2 = ax1_twin.plot(df_valid["Timestamp"], df_valid["GPU_Util"], label="GPU Util [%]", color="darkgreen", linewidth=1.2)
axes1[1].set_ylabel("FPS", color="blue")
ax1_twin.set_ylabel("GPU Load (%)", color="darkgreen")
axes1[1].set_title("Inference Throughput & GPU Utilization")
axes1[1].grid(True, alpha=0.3)
axes1[1].legend(line1 + line2, [l.get_label() for l in line1 + line2], loc="upper right")

# VRAM & Power
ax1_twin2 = axes1[2].twinx()
line3 = axes1[2].plot(df_valid["Timestamp"], df_valid["VRAM_MB"], label="VRAM [MB]", color="crimson", linewidth=1.2)
line4 = ax1_twin2.plot(df_valid["Timestamp"], df_valid["Power_W"], label="Power [W]", color="chocolate", linewidth=1.2)
axes1[2].set_ylabel("VRAM (MB)", color="crimson")
ax1_twin2.set_ylabel("Power (Watt)", color="chocolate")
axes1[2].set_xlabel("Timestamp")
axes1[2].set_title("Resource Footprint (VRAM & Power Draw)")
axes1[2].grid(True, alpha=0.3)
axes1[2].legend(line3 + line4, [l.get_label() for l in line3 + line4], loc="upper right")

plt.tight_layout()
tech_dashboard_filename = latest_file.replace(".csv", "_tech_dashboard.png")
fig1.savefig(tech_dashboard_filename, dpi=300, bbox_inches="tight")
print(f"Technisches Dashboard gespeichert: {tech_dashboard_filename}")


# --- PLOT 2: Separates Sicherheits- und Szenario-Diagramm ---
fig2, ax_impact = plt.subplots(figsize=(10, 5))

categories = ['Normaler Zustand', 'Handynutzung', 'Microsleep (EAR < 0.1)']
meters = [normal_meters_at_130, phone_meters_at_130, microsleep_meters_at_130]
bar_colors = ['#558B2F', '#EF6C00', '#C62828']

bars = ax_impact.barh(categories, meters, color=bar_colors, edgecolor='black', height=0.4)
ax_impact.set_xlabel("Distanz in Metern bei Annahme von 130 km/h (36.1 m/s)", fontsize=10, fontweight="bold")
ax_impact.set_title("Szenarien-Analyse: Zurückgelegte Strecke bei Ablenkung / Inaktivität", fontsize=12, fontweight="bold")
ax_impact.grid(True, axis="x", alpha=0.3)

for bar in bars:
    width = bar.get_width()
    ax_impact.text(width + 2, bar.get_y() + bar.get_height()/2, f'{width:.1f} m', 
                   va='center', ha='left', fontsize=9, fontweight='bold')

summary_text = (
    f"Messwerte (RTX 3050):\n"
    f"• Durchsatz: {avg_fps:.1f} FPS | GPU-Last: {df_valid['GPU_Util'].mean():.1f}% | "
    f"VRAM: {df_valid['VRAM_MB'].mean():.1f} MB | Leistungsaufnahme: {df_valid['Power_W'].mean():.1f} W\n"
    f"• Architektur: Lokale Inferenz (Edge Computing)."
)

fig2.text(
    0.1, 0.02, summary_text,
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#FAFAFA", edgecolor="#BDBDBD", linewidth=1)
)

plt.tight_layout(rect=[0, 0.18, 1, 0.95])
impact_dashboard_filename = latest_file.replace(".csv", "_safety_analysis_chart.png")
fig2.savefig(impact_dashboard_filename, dpi=300, bbox_inches="tight")
print(f"Sicherheits-Analyse-Diagramm gespeichert: {impact_dashboard_filename}")

plt.show()