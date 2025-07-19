import {
  DetectionResult,
  SystemMetrics,
  NetworkGraph,
  AttackType,
  NetworkNode,
  NetworkEdge,
} from "../types";

// Generate a random IP address
const generateRandomIP = () => {
  return `${Math.floor(Math.random() * 255)}.${Math.floor(
    Math.random() * 255
  )}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`;
};

// Generate a random timestamp within the last hour
const generateRandomTimestamp = () => {
  const now = new Date();
  const randomTime = new Date(
    now.getTime() - Math.floor(Math.random() * 3600000)
  );
  return randomTime.toISOString();
};

// Generate random attack types
const attackTypes = [
  "SYN_FLOOD",
  "UDP_FLOOD",
  "HTTP_FLOOD",
  "DNS_AMPLIFICATION",
  "NTP_AMPLIFICATION",
];
const getRandomAttackType = () => {
  return attackTypes[Math.floor(Math.random() * attackTypes.length)];
};

// Generate random detection methods
const detectionMethods = [
  "Autoencoder",
  "Graph Neural Network",
  "Reinforcement Learning",
  "Ensemble",
  "Anomaly Detection",
];
const getRandomDetectionMethod = () => {
  return detectionMethods[Math.floor(Math.random() * detectionMethods.length)];
};

// Generate random protocols
const protocols = ["TCP", "UDP", "HTTP", "ICMP", "DNS"];
const getRandomProtocol = () => {
  return protocols[Math.floor(Math.random() * protocols.length)];
};

// Generate a random detection result
export const generateRandomDetection = (
  isMalicious: boolean = Math.random() > 0.7
): DetectionResult => {
  const threatScore = isMalicious
    ? Math.floor(Math.random() * 30) + 70 // 70-100 for malicious
    : Math.floor(Math.random() * 40) + 10; // 10-50 for benign

  const confidence = isMalicious
    ? Math.random() * 0.3 + 0.7 // 0.7-1.0 for malicious
    : Math.random() * 0.4 + 0.3; // 0.3-0.7 for benign

  // Get attack type based on whether it's malicious
  const attackType = isMalicious
    ? (getRandomAttackType() as AttackType)
    : AttackType.BENIGN;

  return {
    packet_id: `pkt_${Math.random().toString(36).substring(2, 10)}`,
    timestamp: generateRandomTimestamp(),
    src_ip: generateRandomIP(),
    dst_ip: generateRandomIP(),
    src_port: Math.floor(Math.random() * 65535),
    dst_port: Math.floor(Math.random() * 65535),
    is_malicious: isMalicious,
    threat_score: threatScore,
    confidence: confidence,
    detection_method: getRandomDetectionMethod(),
    model_version: "v1.2.3",
    attack_type: attackType,
    protocol: getRandomProtocol(),
    explanation: {
      feature_importance: {
        packet_size: Math.random() * 0.5,
        packet_rate: Math.random() * 0.7,
        entropy: Math.random() * 0.4,
        connection_count: Math.random() * 0.6,
      },
      threshold: 0.75,
      decision_boundary: 0.82,
    },
  };
};

// Generate random system metrics
export const generateRandomMetrics = (): SystemMetrics => {
  return {
    timestamp: generateRandomTimestamp(),
    packets_processed: Math.floor(Math.random() * 100000) + 10000,
    malicious_packets: Math.floor(Math.random() * 1000) + 100,
    total_detections: Math.floor(Math.random() * 2000) + 200,
    cpu_usage: Math.random() * 80 + 10, // 10-90%
    memory_usage: Math.random() * 70 + 15, // 15-85%
    active_connections: Math.floor(Math.random() * 500) + 50,
    processing_latency_ms: Math.floor(Math.random() * 40) + 2, // 2-42ms
    threat_level: Math.floor(Math.random() * 5) + 1, // 1-5
  };
};

// Generate a random network graph
export const generateRandomNetworkGraph = (): NetworkGraph => {
  const nodeCount = Math.floor(Math.random() * 20) + 10; // 10-30 nodes
  const nodes: NetworkNode[] = [];
  const edges: NetworkEdge[] = [];

  // Generate timestamps for first and last seen
  const now = new Date();

  // Generate nodes
  for (let i = 0; i < nodeCount; i++) {
    const isMalicious = Math.random() > 0.7;
    const threatScore = isMalicious
      ? Math.floor(Math.random() * 30) + 70 // 70-100 for malicious
      : Math.floor(Math.random() * 40) + 10; // 10-50 for benign

    const firstSeen = new Date(
      now.getTime() - Math.floor(Math.random() * 86400000)
    ); // Up to 24 hours ago
    const lastSeen = new Date(
      firstSeen.getTime() +
        Math.floor(Math.random() * (now.getTime() - firstSeen.getTime()))
    );

    const ipAddress = generateRandomIP();

    nodes.push({
      node_id: `node_${i}`,
      ip_address: ipAddress,
      is_malicious: isMalicious,
      threat_score: threatScore,
      packet_count: Math.floor(Math.random() * 1000) + 50,
      byte_count: Math.floor(Math.random() * 1000000) + 10000,
      connection_count: Math.floor(Math.random() * 10) + 1,
      first_seen: firstSeen.toISOString(),
      last_seen: lastSeen.toISOString(),
    });
  }

  // Generate edges (connections between nodes)
  const edgeCount = Math.floor(Math.random() * 30) + 20; // 20-50 edges
  for (let i = 0; i < edgeCount; i++) {
    const sourceIndex = Math.floor(Math.random() * nodes.length);
    let targetIndex;
    do {
      targetIndex = Math.floor(Math.random() * nodes.length);
    } while (targetIndex === sourceIndex);

    const sourceNode = nodes[sourceIndex];
    const targetNode = nodes[targetIndex];

    edges.push({
      edge_id: `edge_${i}`,
      source_ip: sourceNode.ip_address,
      target_ip: targetNode.ip_address,
      flow_count: Math.floor(Math.random() * 100) + 1,
      total_bytes: Math.floor(Math.random() * 1000000) + 1000,
      avg_packet_size: Math.floor(Math.random() * 1000) + 100,
      connection_duration: Math.floor(Math.random() * 3600) + 60, // 1-60 minutes
      protocols: [getRandomProtocol()],
    });
  }

  return {
    nodes,
    edges,
    timestamp: new Date().toISOString(),
  };
};

// Generate a batch of random detections
export const generateRandomDetections = (
  count: number = 50
): DetectionResult[] => {
  const detections = [];
  const maliciousCount = Math.floor(count * 0.3); // 30% malicious

  // Generate malicious detections
  for (let i = 0; i < maliciousCount; i++) {
    detections.push(generateRandomDetection(true));
  }

  // Generate benign detections
  for (let i = 0; i < count - maliciousCount; i++) {
    detections.push(generateRandomDetection(false));
  }

  // Sort by timestamp (newest first)
  return detections.sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );
};
