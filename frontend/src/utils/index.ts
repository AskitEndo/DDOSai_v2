import { AttackType } from "../types";

/**
 * Format a number with thousands separators
 */
export const formatNumber = (num: number): string => {
  return num.toLocaleString();
};

/**
 * Format bytes to human-readable format (KB, MB, GB)
 */
export const formatBytes = (bytes: number): string => {
  if (bytes === 0) return "0 Bytes";

  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
};

/**
 * Format timestamp to readable format
 */
export const formatTimestamp = (timestamp: string): string => {
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
};

/**
 * Get color based on threat level (0-100)
 */
export const getThreatLevelColor = (level: number): string => {
  if (level >= 80) return "text-danger";
  if (level >= 60) return "text-warning";
  if (level >= 40) return "text-yellow-500";
  return "text-success";
};

/**
 * Get color for attack type
 */
export const getAttackTypeColor = (attackType: string): string => {
  switch (attackType) {
    case AttackType.SYN_FLOOD:
      return "text-red-500";
    case AttackType.UDP_FLOOD:
      return "text-orange-500";
    case AttackType.HTTP_FLOOD:
      return "text-yellow-500";
    case AttackType.SLOWLORIS:
      return "text-purple-500";
    case AttackType.BENIGN:
      return "text-green-500";
    default:
      return "text-gray-400";
  }
};

/**
 * Get display name for attack type
 */
export const getAttackTypeDisplayName = (attackType: string): string => {
  switch (attackType) {
    case AttackType.SYN_FLOOD:
      return "SYN Flood";
    case AttackType.UDP_FLOOD:
      return "UDP Flood";
    case AttackType.HTTP_FLOOD:
      return "HTTP Flood";
    case AttackType.SLOWLORIS:
      return "Slowloris";
    case AttackType.BENIGN:
      return "Benign";
    default:
      return "Unknown";
  }
};

/**
 * Truncate text with ellipsis
 */
export const truncateText = (text: string, maxLength: number): string => {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + "...";
};
