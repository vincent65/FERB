#!/usr/bin/env python3
"""
Parse NCCL topology XML (with NVLink information) to PKB JSON format.

This parser handles NCCL XML that includes:
- GPU elements with dev/rank/sm attributes
- NVLink connection information
- PCIe hierarchy and NUMA topology
- Network interface cards

Usage:
    python parse_nccl_topo.py <nccl_topo.xml>
"""

import xml.etree.ElementTree as ET
import json
import sys
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict


# GPU device IDs
GPU_DEVICES = {
    "0x2330": "NVIDIA H100 80GB HBM3",
    "0x2331": "NVIDIA H100 PCIe",
    "0x20b0": "NVIDIA A100-SXM4-40GB",
    "0x20b2": "NVIDIA A100-SXM4-80GB",
    "0x20f1": "NVIDIA A100-PCIe-80GB",
    "0x1db6": "NVIDIA V100-SXM2-32GB",
    "0x1db8": "NVIDIA V100-PCIe-32GB",
}

def get_gpu_specs(device_id: str) -> Dict[str, Any]:
    """Get GPU specifications from device ID."""
    name = GPU_DEVICES.get(device_id, f"NVIDIA GPU (device {device_id})")
    
    specs = {
        "name": name,
        "memory_gb": 80,
        "memory_bandwidth_gb_s": 1000,
        "compute_capability": "8.0",
        "nvlink_version": "3.0",
        "nvlink_bw_per_link": 25,  # GB/s per lane
    }
    
    if "H100" in name:
        specs.update({
            "memory_gb": 80,
            "memory_bandwidth_gb_s": 3352,
            "compute_capability": "9.0",
            "nvlink_version": "4.0",
            "nvlink_bw_per_link": 25,  # NVLink 4.0: ~25 GB/s per lane (18 lanes total)
        })
    elif "A100" in name:
        specs["memory_gb"] = 80 if "80GB" in name else 40
        specs["memory_bandwidth_gb_s"] = 2039 if "80GB" in name else 1555
        specs["compute_capability"] = "8.0"
        specs["nvlink_version"] = "3.0"
        specs["nvlink_bw_per_link"] = 25  # NVLink 3.0
    elif "V100" in name:
        specs.update({
            "memory_gb": 32,
            "memory_bandwidth_gb_s": 900,
            "compute_capability": "7.0",
            "nvlink_version": "2.0",
            "nvlink_bw_per_link": 25,
        })
    
    return specs


def parse_pcie_speed(speed_str: str) -> Tuple[int, int]:
    """Parse PCIe speed string to generation and bandwidth."""
    if "32.0 GT/s" in speed_str:
        return 5, 128  # Gen 5 x16
    elif "16.0 GT/s" in speed_str:
        return 4, 64   # Gen 4 x16
    elif "8.0 GT/s" in speed_str:
        return 3, 32   # Gen 3 x16
    return 4, 64


def find_gpus_recursive(elem: ET.Element, numa_id: int, 
                        gpu_list: List[Dict], busid_to_gpu: Dict[str, int]) -> None:
    """Recursively find all GPU elements in the PCIe tree."""
    
    # Check if this is a GPU element
    gpu_elem = elem.find("gpu")
    if gpu_elem is not None:
        dev = int(gpu_elem.get("dev", -1))
        rank = int(gpu_elem.get("rank", dev))
        sm = gpu_elem.get("sm", "90")
        gdr = gpu_elem.get("gdr", "1")
        
        # Get parent PCI element info
        busid = elem.get("busid", "")
        device_id = elem.get("device", "0x2330")
        link_speed = elem.get("link_speed", "32.0 GT/s PCIe")
        
        pcie_gen, pcie_bw = parse_pcie_speed(link_speed)
        specs = get_gpu_specs(device_id)
        
        # Parse NVLink connections
        nvlink_targets = []
        total_nvlink_lanes = 0
        for nvlink in gpu_elem.findall("nvlink"):
            target = nvlink.get("target", "")
            count = int(nvlink.get("count", 0))
            nvlink_targets.append({
                "target_busid": target,
                "lanes": count
            })
            total_nvlink_lanes += count
        
        gpu_info = {
            "id": dev,
            "rank": rank,
            "name": specs["name"],
            "compute_capability": specs["compute_capability"],
            "memory_gb": specs["memory_gb"],
            "memory_bandwidth_gb_s": specs["memory_bandwidth_gb_s"],
            "pci_bus_id": busid,
            "pcie_gen": pcie_gen,
            "pcie_lanes": 16,
            "pcie_bandwidth_gb_s": pcie_bw,
            "numa_node": numa_id,
            "cpu_affinity": [numa_id],
            "device_id": device_id,
            "nvlink_raw": nvlink_targets,
            "nvlink_total_lanes": total_nvlink_lanes,
            "gdr_support": gdr == "1",
        }
        
        gpu_list.append(gpu_info)
        busid_to_gpu[busid] = dev
        return
    
    # Recurse into child PCI elements
    for child in elem.findall("pci"):
        find_gpus_recursive(child, numa_id, gpu_list, busid_to_gpu)


def analyze_nvlink_topology(gpus: List[Dict]) -> Dict[str, Any]:
    """Analyze NVLink connections to determine topology type."""
    
    if not gpus:
        return {"type": "unknown", "has_nvswitch": False}
    
    num_gpus = len(gpus)
    
    # Check for NVSwitch indicators:
    # 1. Very high lane counts (18 lanes = all NVLink to NVSwitch)
    # 2. Same target addresses across GPUs (NVSwitch ports)
    # 3. Invalid/special target addresses
    
    nvswitch_indicators = 0
    target_addresses = defaultdict(int)
    
    for gpu in gpus:
        lanes = gpu["nvlink_total_lanes"]
        
        # H100 has 18 NVLink lanes total
        if lanes >= 18:
            nvswitch_indicators += 1
        
        # Collect target addresses
        for nvlink in gpu["nvlink_raw"]:
            target = nvlink["target_busid"]
            if "fffffff" in target.lower() or "ffffff" in target.lower():
                # Invalid address = likely NVSwitch fabric
                nvswitch_indicators += 1
            target_addresses[target] += 1
    
    # If multiple GPUs target the same addresses, likely NVSwitch ports
    common_targets = [addr for addr, count in target_addresses.items() if count > 1]
    
    has_nvswitch = (
        nvswitch_indicators > 0 or 
        len(common_targets) > 0 or
        (num_gpus == 8 and "H100" in gpus[0]["name"])
    )
    
    if has_nvswitch:
        topology_type = "nvswitch_all_to_all"
        notes = f"NVSwitch fabric detected. All {num_gpus} GPUs have all-to-all connectivity."
    else:
        topology_type = "direct_nvlink"
        notes = f"Direct NVLink connections between GPUs."
    
    return {
        "type": topology_type,
        "has_nvswitch": has_nvswitch,
        "notes": notes,
        "nvswitch_ports": common_targets if has_nvswitch else []
    }


def build_connectivity(gpus: List[Dict], topology_info: Dict, busid_to_gpu: Dict[str, int]) -> None:
    """Build GPU connectivity information."""
    
    num_gpus = len(gpus)
    
    if topology_info["has_nvswitch"]:
        # NVSwitch: All-to-all connectivity
        for gpu in gpus:
            gpu["nvlink_peers"] = [i for i in range(num_gpus) if i != gpu["id"]]
            # Estimate bandwidth (H100 with NVSwitch: ~900 GB/s bidirectional)
            specs = get_gpu_specs(gpu["device_id"])
            lanes = gpu["nvlink_total_lanes"]
            gpu["nvlink_bandwidth_gb_s"] = int(lanes * specs["nvlink_bw_per_link"])
    else:
        # Direct NVLink: Resolve target bus IDs to GPU IDs
        for gpu in gpus:
            nvlink_peers = []
            peer_lanes = defaultdict(int)
            
            # Resolve each NVLink target to a GPU ID
            for nvlink in gpu["nvlink_raw"]:
                target_busid = nvlink["target_busid"]
                lanes = nvlink["lanes"]
                
                # Try to map bus ID to GPU ID
                if target_busid in busid_to_gpu:
                    peer_gpu_id = busid_to_gpu[target_busid]
                    if peer_gpu_id != gpu["id"]:  # Don't add self
                        peer_lanes[peer_gpu_id] += lanes
            
            # Store peers sorted by ID
            nvlink_peers = sorted(peer_lanes.keys())
            gpu["nvlink_peers"] = nvlink_peers
            
            # Calculate bandwidth based on total lanes
            specs = get_gpu_specs(gpu["device_id"])
            lanes = gpu["nvlink_total_lanes"]
            gpu["nvlink_bandwidth_gb_s"] = int(lanes * specs["nvlink_bw_per_link"]) if lanes > 0 else 0
            
            # Store per-peer lane counts for detailed analysis
            gpu["nvlink_peer_lanes"] = dict(peer_lanes)
    
    # All GPUs can communicate (even if through PCIe)
    for gpu in gpus:
        gpu["connected_to"] = [i for i in range(num_gpus) if i != gpu["id"]]


def build_interconnects(gpus: List[Dict], topology_info: Dict) -> List[Dict]:
    """Build interconnect list."""
    
    interconnects = []
    num_gpus = len(gpus)
    
    if num_gpus == 0:
        return interconnects
    
    specs = get_gpu_specs(gpus[0]["device_id"])
    
    if topology_info["has_nvswitch"]:
        # NVSwitch: One entry showing all-to-all
        # Calculate bandwidth based on lane count
        avg_lanes = sum(g["nvlink_total_lanes"] for g in gpus) / len(gpus)
        bw = int(avg_lanes * specs["nvlink_bw_per_link"])
        
        interconnects.append({
            "type": "NVSwitch",
            "version": specs["nvlink_version"],
            "gpus": list(range(num_gpus)),
            "bidirectional_bandwidth_gb_s": bw,
            "notes": "All-to-all connectivity via NVSwitch fabric"
        })
    else:
        # Direct NVLink: Create entries for each connection
        seen_pairs = set()
        
        for gpu in gpus:
            gpu_id = gpu["id"]
            
            # Get per-peer lane information if available
            peer_lanes = gpu.get("nvlink_peer_lanes", {})
            
            for peer_id in gpu["nvlink_peers"]:
                # Avoid duplicate entries (bidirectional)
                pair = tuple(sorted([gpu_id, peer_id]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                
                # Calculate bandwidth for this connection
                lanes = peer_lanes.get(peer_id, 0)
                bw = int(lanes * specs["nvlink_bw_per_link"]) if lanes > 0 else 300  # Default
                
                interconnects.append({
                    "type": "NVLink",
                    "version": specs["nvlink_version"],
                    "gpus": list(pair),
                    "lanes": lanes,
                    "bidirectional_bandwidth_gb_s": bw,
                    "notes": f"Direct NVLink connection ({lanes} lanes)"
                })
    
    return interconnects


def parse_nccl_topology(xml_file: str) -> Dict[str, Any]:
    """Parse NCCL topology XML to PKB JSON format."""
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception as e:
        return {"error": f"Failed to parse XML: {e}"}
    
    gpus = []
    cpus = []
    busid_to_gpu = {}
    
    # Parse each CPU/NUMA node
    for cpu_elem in root.findall("cpu"):
        numa_id = int(cpu_elem.get("numaid", 0))
        arch = cpu_elem.get("arch", "unknown")
        vendor = cpu_elem.get("vendor", "unknown")
        
        # Find all GPUs under this CPU
        gpus_before = len(gpus)
        for pci_elem in cpu_elem.findall("pci"):
            find_gpus_recursive(pci_elem, numa_id, gpus, busid_to_gpu)
        gpus_after = len(gpus)
        
        # Record CPU info
        connected_gpus = list(range(gpus_before, gpus_after))
        cpus.append({
            "id": numa_id,
            "cores": 64,  # Typical for server systems
            "arch": arch,
            "vendor": vendor,
            "connected_gpus": connected_gpus
        })
    
    if not gpus:
        return {"error": "No GPUs found in XML"}
    
    # Sort GPUs by device ID
    gpus.sort(key=lambda g: g["id"])
    
    num_gpus = len(gpus)
    
    # Analyze NVLink topology
    topology_info = analyze_nvlink_topology(gpus)
    
    # Build connectivity
    build_connectivity(gpus, topology_info, busid_to_gpu)
    
    # Build interconnects
    interconnects = build_interconnects(gpus, topology_info)
    
    # Detect system type
    gpu_model = gpus[0]["name"]
    if "H100" in gpu_model:
        system_name = f"H100 System ({num_gpus} GPUs)"
    elif "A100" in gpu_model:
        system_name = f"A100 System ({num_gpus} GPUs)"
    else:
        system_name = f"Multi-GPU System ({num_gpus} GPUs)"
    
    # Build topology notes
    topology_notes = (
        f"Parsed from NCCL topology XML. "
        f"{num_gpus}x {gpu_model} across {len(cpus)} NUMA nodes. "
        f"{topology_info['notes']}"
    )
    
    # Add helpful context for LLMs
    if topology_info["has_nvswitch"]:
        optimal_patterns = [
            "All-reduce: Ring or recursive doubling (all paths have equal bandwidth)",
            "All-to-all: Direct transfers (no intermediate hops)",
            "Point-to-point: Direct NVSwitch connection with full bandwidth"
        ]
    else:
        # Check if there's any NVLink at all
        has_nvlink = any(gpu["nvlink_total_lanes"] > 0 for gpu in gpus)
        if has_nvlink:
            optimal_patterns = [
                "All-reduce: Ring algorithm recommended",
                "Communication patterns depend on NVLink topology",
                "Some GPU pairs may require routing through intermediates or use PCIe"
            ]
        else:
            optimal_patterns = [
                "All communication over PCIe (no NVLink available)",
                "All-reduce: Ring algorithm to minimize PCIe bottlenecks",
                "Consider minimizing communication volume"
            ]
    
    # Build warnings list
    warnings = []
    
    # Check for unresolved NVLink targets
    if not topology_info["has_nvswitch"]:
        for gpu in gpus:
            for nvlink in gpu.get("nvlink_raw", []):
                target = nvlink["target_busid"]
                if target not in busid_to_gpu and "fffffff" not in target.lower():
                    warnings.append(
                        f"GPU {gpu['id']}: NVLink target {target} could not be resolved to a GPU ID"
                    )
    
    # Check for unknown GPU models
    for gpu in gpus:
        if gpu["device_id"] not in GPU_DEVICES:
            warnings.append(
                f"GPU {gpu['id']}: Unknown device ID {gpu['device_id']}, using estimated specs"
            )
    
    # Check for asymmetric topologies
    if not topology_info["has_nvswitch"]:
        peer_counts = [len(gpu.get("nvlink_peers", [])) for gpu in gpus]
        if len(set(peer_counts)) > 1:
            warnings.append(
                "Asymmetric NVLink topology detected: Different GPUs have different numbers of NVLink peers"
            )
    
    topology = {
        "system_name": system_name,
        "num_nodes": 1,
        "num_gpus": num_gpus,
        "gpus": gpus,
        "cpus": cpus,
        "interconnects": interconnects,
        "topology_type": topology_info["type"],
        "has_nvswitch": topology_info["has_nvswitch"],
        "topology_notes": topology_notes,
        "optimal_patterns": optimal_patterns,
    }
    
    if warnings:
        topology["warnings"] = warnings
    
    return topology


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_nccl_topo.py <nccl_topo.xml>")
        print()
        print("Parses NCCL topology XML (with NVLink info) to LLM-friendly JSON")
        print()
        print("To generate this XML, set NCCL_TOPO_DUMP_FILE before running")
        print("any NCCL program:")
        print("  export NCCL_TOPO_DUMP_FILE=topo.xml")
        print("  python -m torch.distributed.launch ...")
        sys.exit(1)
    
    xml_file = sys.argv[1]
    
    print("=" * 70)
    print("NCCL Topology Parser for ParallelKernelBench")
    print("=" * 70)
    print(f"\nParsing: {xml_file}\n")
    
    topology = parse_nccl_topology(xml_file)
    
    if "error" in topology:
        print(f"❌ Error: {topology['error']}")
        sys.exit(1)
    
    # Print warnings first if any
    if "warnings" in topology:
        print("⚠️  WARNINGS:")
        for warning in topology["warnings"]:
            print(f"  - {warning}")
        print()
    
    # Print summary
    print(f"✓ System: {topology['system_name']}")
    print(f"✓ GPUs: {topology['num_gpus']}")
    print(f"✓ NUMA nodes: {len(topology['cpus'])}")
    print(f"✓ Topology: {topology['topology_type']}")
    
    if topology["has_nvswitch"]:
        print("✓ NVSwitch: Detected (all-to-all connectivity)")
    
    print(f"\nGPU Details:")
    for gpu in topology["gpus"]:
        lanes = gpu["nvlink_total_lanes"]
        bw = gpu.get("nvlink_bandwidth_gb_s", 0)
        print(f"  GPU {gpu['id']}: {gpu['name']}")
        print(f"    - NUMA: {gpu['numa_node']}, PCIe: {gpu['pci_bus_id']}")
        print(f"    - NVLink: {lanes} lanes, {bw} GB/s bandwidth")
        if gpu.get("nvlink_peers"):
            peers = gpu['nvlink_peers']
            print(f"    - Peers: GPU {peers}")
    
    print(f"\nInterconnects: {len(topology['interconnects'])} connections")
    if not topology["has_nvswitch"] and topology["interconnects"]:
        # Show first few direct NVLink connections
        for ic in topology["interconnects"][:5]:
            if ic["type"] == "NVLink":
                print(f"  - GPU {ic['gpus'][0]} ↔ GPU {ic['gpus'][1]}: "
                      f"{ic['lanes']} lanes, {ic['bidirectional_bandwidth_gb_s']} GB/s")
        if len(topology["interconnects"]) > 5:
            print(f"  ... and {len(topology['interconnects']) - 5} more")
    
    # Save JSON
    output_file = xml_file.replace(".xml", "_parsed.json")
    with open(output_file, "w") as f:
        json.dump(topology, f, indent=2)
    
    print(f"\n✓ Saved to: {output_file}")
    print("\n✅ Parser supports:")
    print("  ✓ NVSwitch all-to-all topologies (DGX H100)")
    print("  ✓ Direct NVLink topologies (DGX A100)")
    print("  ✓ PCIe-only systems (workstations)")
    print("  ✓ Unknown GPU models (with warnings)")
    print("  ✓ Asymmetric topologies")


if __name__ == "__main__":
    main()

