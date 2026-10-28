import numpy as np
import pandas as pd


SPEC_SR650 = "https://www.spec.org/power_ssj2008/results/res2023q3/power_ssj2008-20230619-01282.html"
SPEC_6448Y = "https://www.spec.org/power_ssj2008/results/res2024q1/power_ssj2008-20240227-01377.html"
H100 = "https://lenovopress.lenovo.com/lp1732-thinksystem-nvidia-h100-pcie-gen5-gpu"
DGX = "https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html"
GB200 = "https://docs.nvidia.com/dgx/dgxgb200-user-guide/hardware.html"
BLACKWELL = "https://www.supermicro.com/en/pressreleases/supermicro-grows-ai-optimized-product-portfolio-new-generation-systems-and-rack"
GRACE = "https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/power-thermals.html"
HGX = "https://arxiv.org/html/2412.08602v2"
LOAD = np.arange(11) / 10
SR650_W = np.array([163, 280, 322, 363, 405, 453, 508, 574, 674, 808, 915.0])
DELL_W = np.array([76.2, 122, 133, 149, 164, 179, 253, 290, 315, 318, 318.0])
SHAPES = {"sr650": (SR650_W - SR650_W[0]) / (SR650_W[-1] - SR650_W[0]),
          "6448y": (DELL_W - DELL_W[0]) / (DELL_W[-1] - DELL_W[0])}
CASES = ("low", "central", "high")

# 顺序为 P_base、每颗 CPU 空载/满载、每张 GPU 空载/满载，单位 W。
PARAMETERS = {
    "C20": {"low": (100, 10, 180, 0, 0), "central": (150, 20, 225, 0, 0), "high": (250, 35, 270, 0, 0)},
    "C21": {"low": (150, 10, 180, 25, 280), "central": (200, 20, 225, 50, 350), "high": (350, 35, 270, 80, 350)},
    "C22": {"low": (200, 10, 180, 25, 280), "central": (250, 20, 225, 50, 350), "high": (450, 35, 270, 80, 350)},
    "C28": {"low": (1000, 25, 280, 40, 560), "central": (1800, 50, 350, 70, 700), "high": (3000, 75, 400, 120, 700)},
    "R-GB200": {"low": (18000, 20, 180, 80, 960), "central": (24600, 40, 250, 120, 1200), "high": (30000, 80, 350, 200, 1200)},
}


def validate_utilization(value):
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
        raise ValueError("利用率代理必须为 0 至 1 的有限数值")
    return value


def component_power(configuration, u_cpu, u_gpu, case="central"):
    u_cpu, u_gpu = validate_utilization(u_cpu), validate_utilization(u_gpu)
    key = configuration["configuration_id"]
    base, cpu_idle, cpu_full, gpu_idle, gpu_full = PARAMETERS[key][case]
    if key == "R-GB200":
        cpu_shape = u_cpu
    else:
        same_cpu = np.interp(u_cpu, LOAD, SHAPES["6448y"])
        same_platform = np.interp(u_cpu, LOAD, SHAPES["sr650"])
        cpu_shape = {"low": np.minimum(same_cpu, same_platform), "central": same_cpu,
                     "high": np.maximum(same_cpu, same_platform)}[case]
    gpu_shape = u_gpu ** {"low": 1.2, "central": 1.0, "high": 0.8}[case]
    cpu = configuration["cpu_packages_per_unit"] * (cpu_idle + (cpu_full - cpu_idle) * cpu_shape)
    gpu = configuration["gpu_devices_per_unit"] * (gpu_idle + (gpu_full - gpu_idle) * gpu_shape)
    return {"base_w": np.broadcast_to(float(base), np.shape(cpu)), "cpu_w": cpu, "gpu_w": gpu,
            "unit_w": base + cpu + gpu}


def reference_power(configuration):
    return float(component_power(configuration, 1, 1)["unit_w"])


def parameter_table(configurations):
    rows = []
    for config in configurations:
        key = config["configuration_id"]
        for case in CASES:
            base, ci, cf, gi, gf = PARAMETERS[key][case]
            full = float(component_power(config, 1, 1, case)["unit_w"])
            idle = float(component_power(config, 0, 0, case)["unit_w"])
            rows.append({"configuration_id": key, "case": case, "unit": config["research_unit"],
                "cpu_packages": config["cpu_packages_per_unit"], "gpus": config["gpu_devices_per_unit"],
                "p_base_w": base, "cpu_idle_w_each": ci, "cpu_full_w_each": cf,
                "gpu_idle_w_each": gi, "gpu_full_w_each": gf, "unit_idle_w": idle, "unit_full_w": full,
                "sizing_reference_w": reference_power(config), "units_at_10mw": int(10e6 // reference_power(config)),
                "power_status": "assumption; public_specifications_and_transferred_benchmark_shape",
                "base_status": "assumption; memory_fans_network_storage_switches_PSU_loss_lumped_constant",
                "cpu_endpoints_status": "assumption; 6448Y_central_full_equals_vendor_TDP" if key in ("C20", "C21", "C22") else "assumption",
                "gpu_idle_status": "not_applicable" if key == "C20" else "assumption; no_matching_isolated_idle_measurement",
                "gpu_full_status": "not_applicable" if key == "C20" else "assumed_operating_endpoint; vendor_TDP_ceiling",
                "cpu_shape_source": GRACE if key == "R-GB200" else f"{SPEC_6448Y};{SPEC_SR650}",
                "gpu_ceiling_source": BLACKWELL if key == "R-GB200" else H100,
                "platform_source": config.get("hardware_source_url") or config["source_url"],
                "external_measurement_comparison": HGX if key == "C28" else "",
                "host_cpu_per_gpu_fraction": {"low": .10, "central": .25, "high": .40}[case],
                "gpu_exponent": {"low": 1.2, "central": 1., "high": .8}[case]})
    return pd.DataFrame(rows)


def evidence_table():
    rows = []
    for name, watts, url, cpu, count in (("SR650 V3", SR650_W, SPEC_SR650, "Xeon Platinum 8490H", 2),
                                        ("PowerEdge HS5610", DELL_W, SPEC_6448Y, "Xeon Gold 6448Y", 1)):
        for load, power in zip(LOAD, watts):
            rows.append({"platform": name, "cpu_model": cpu, "cpu_packages": count, "target_load_fraction": load,
                "whole_system_ac_w": power, "status": "measured_SPECpower_ssj2008_whole_system_not_CPU_package",
                "source_url": url})
    return pd.DataFrame(rows)
