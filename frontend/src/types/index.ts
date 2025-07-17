// API Response Types
export interface ApiResponse<T> {
  data: T;
  status: number;
  message?: string;
}

// Traffic Data Types
export interface TrafficPacket {
  timestamp: string;
  packet_id: string;
  src_ip: string;
  dst_ip: string;
  src_port: number;
  dst_port: number;
  protocol: string;
  packet_size: number;
  ttl: number;
  flags: string[];
  payload_entropy: number;
}

export interface NetworkFlow {
  flow_id: string;
  src_ip: string;
  dst_ip: string;
  src_port: number;
  dst_port: number;
  protocol: string;
  start_time: string;
  end_time: string;
  packet_count: number;
  byte_count: number;
  avg_packet_size: number;
  flow_duration: number;
}

// Detection Result Types
export enum AttackType {
  BENIGN = "benign",
  SYN_FLOOD = "syn_flood",
  UDP_FLOOD = "udp_flood",
  HTTP_FLOOD = "http_flood",
  SLOWLORIS = "slowloris",
  UNKNOWN = "unknown",
}

export interface DetectionResult {
  timestamp: string;
  packet_id: string;
  flow_id?: string;
  src_ip: string;
  dst_ip: string;
  src_port?: number;
  dst_port?: number;
  protocol?: string;
  is_malicious: boolean;
  threat_score: number;
  attack_type: AttackType;
  detection_method: string;
  confidence: number;
  explanation: Record<string, any>;
  model_version: string;
}

// Network Graph Types
export interface NetworkNode {
  node_id: string;
  ip_address: string;
  packet_count: number;
  byte_count: number;
  connection_count: number;
  threat_score: number;
  is_malicious: boolean;
  first_seen: string;
  last_seen: string;
}

export interface NetworkEdge {
  edge_id: string;
  source_ip: string;
  target_ip: string;
  flow_count: number;
  total_bytes: number;
  avg_packet_size: number;
  connection_duration: number;
  protocols: string[];
}

export interface NetworkGraph {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  timestamp: string;
}

// System Metrics Types
export interface SystemMetrics {
  timestamp: string;
  packets_processed: number;
  processing_latency_ms: number;
  cpu_usage: number;
  memory_usage: number;
  active_connections: number;
  threat_level: number;
  malicious_packets: number;
  total_detections: number;
}

// Simulation Types
export interface SimulationConfig {
  attack_type: string;
  target_ip: string;
  target_port: number;
  duration: number;
  packet_rate: number;
  packet_size?: number;
  num_threads?: number;
  num_connections?: number;
  connection_rate?: number;
  use_https?: boolean;
}

export interface SimulationStatus {
  status: "idle" | "running" | "paused" | "completed" | "error";
  simulation_id?: string;
  attack_type?: string;
  target_ip?: string;
  target_port?: number;
  packets_sent: number;
  bytes_sent: number;
  start_time?: string;
  end_time?: string;
  duration: number;
  packet_rate: number;
  errors: number;
  elapsed_time?: number;
  current_packet_rate?: number;
}

// WebSocket Message Types
export interface WebSocketMessage {
  type: "detection" | "alert" | "metrics" | "graph" | "simulation";
  data: any;
}

// Theme Types
export type ThemeMode = "dark" | "light";

// Dashboard Panel Types
export interface DashboardPanel {
  id: string;
  title: string;
  type: "traffic" | "threats" | "graph" | "metrics" | "xai" | "logs";
  width: "full" | "half" | "third";
  height: "small" | "medium" | "large";
}

// User Settings
export interface UserSettings {
  theme: ThemeMode;
  refreshRate: number;
  alertSound: boolean;
  dashboardLayout: DashboardPanel[];
}
