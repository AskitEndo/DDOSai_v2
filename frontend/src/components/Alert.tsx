import React, { useState, useEffect } from "react";
import { AlertTriangle, CheckCircle, Info, X, AlertCircle } from "lucide-react";

export type AlertType = "success" | "error" | "warning" | "info";

interface AlertProps {
  type: AlertType;
  message: string;
  autoClose?: boolean;
  duration?: number;
  onClose?: () => void;
}

const Alert: React.FC<AlertProps> = ({
  type,
  message,
  autoClose = true,
  duration = 5000,
  onClose,
}) => {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    if (autoClose) {
      const timer = setTimeout(() => {
        setVisible(false);
        onClose?.();
      }, duration);

      return () => clearTimeout(timer);
    }
  }, [autoClose, duration, onClose]);

  const handleClose = () => {
    setVisible(false);
    onClose?.();
  };

  if (!visible) return null;

  const alertConfig = {
    success: {
      icon: CheckCircle,
      bgColor: "bg-success/10",
      borderColor: "border-success/30",
      textColor: "text-success",
    },
    error: {
      icon: AlertCircle,
      bgColor: "bg-danger/10",
      borderColor: "border-danger/30",
      textColor: "text-danger",
    },
    warning: {
      icon: AlertTriangle,
      bgColor: "bg-warning/10",
      borderColor: "border-warning/30",
      textColor: "text-warning",
    },
    info: {
      icon: Info,
      bgColor: "bg-primary/10",
      borderColor: "border-primary/30",
      textColor: "text-primary",
    },
  };

  const { icon: Icon, bgColor, borderColor, textColor } = alertConfig[type];

  return (
    <div
      className={`flex items-center p-4 rounded-md ${bgColor} border ${borderColor} mb-4`}
    >
      <Icon className={`w-5 h-5 ${textColor} mr-3`} />
      <div className="flex-1 text-white">{message}</div>
      <button
        onClick={handleClose}
        className="text-gray-400 hover:text-white transition-colors"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
};

export default Alert;
