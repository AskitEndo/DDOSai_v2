import { io, Socket } from "socket.io-client";
import { WebSocketMessage } from "../types";

class WebSocketService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();

  constructor() {
    this.connect();
  }

  private connect() {
    const wsUrl = import.meta.env.VITE_WS_URL || "ws://localhost:8000";

    this.socket = io(wsUrl, {
      transports: ["websocket"],
      autoConnect: true,
    });

    this.socket.on("connect", () => {
      console.log("WebSocket connected");
      this.reconnectAttempts = 0;
    });

    this.socket.on("disconnect", () => {
      console.log("WebSocket disconnected");
      this.handleReconnect();
    });

    this.socket.on("connect_error", (error) => {
      console.error("WebSocket connection error:", error);
      this.handleReconnect();
    });

    // Handle incoming messages
    this.socket.on("message", (message: WebSocketMessage) => {
      this.handleMessage(message);
    });

    // Handle specific event types
    this.socket.on("detection", (data) => {
      this.emit("detection", data);
    });

    this.socket.on("alert", (data) => {
      this.emit("alert", data);
    });

    this.socket.on("metrics", (data) => {
      this.emit("metrics", data);
    });

    this.socket.on("graph", (data) => {
      this.emit("graph", data);
    });

    this.socket.on("simulation", (data) => {
      this.emit("simulation", data);
    });
  }

  private handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(
        `Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`
      );

      setTimeout(() => {
        this.connect();
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      console.error("Max reconnection attempts reached");
    }
  }

  private handleMessage(message: WebSocketMessage) {
    this.emit(message.type, message.data);
  }

  private emit(event: string, data: any) {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach((callback) => callback(data));
    }
  }

  public on(event: string, callback: (data: any) => void) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);

    // Return unsubscribe function
    return () => {
      const eventListeners = this.listeners.get(event);
      if (eventListeners) {
        eventListeners.delete(callback);
        if (eventListeners.size === 0) {
          this.listeners.delete(event);
        }
      }
    };
  }

  public off(event: string, callback?: (data: any) => void) {
    if (callback) {
      const eventListeners = this.listeners.get(event);
      if (eventListeners) {
        eventListeners.delete(callback);
        if (eventListeners.size === 0) {
          this.listeners.delete(event);
        }
      }
    } else {
      this.listeners.delete(event);
    }
  }

  public send(event: string, data: any) {
    if (this.socket && this.socket.connected) {
      this.socket.emit(event, data);
    } else {
      console.warn("WebSocket not connected, cannot send message");
    }
  }

  public disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.listeners.clear();
  }

  public isConnected(): boolean {
    return this.socket?.connected || false;
  }
}

// Create singleton instance
const webSocketService = new WebSocketService();

export default webSocketService;
