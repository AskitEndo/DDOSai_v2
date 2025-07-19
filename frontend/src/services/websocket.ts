import { WebSocketMessage } from "../types";

class WebSocketService {
  private socket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();

  constructor() {
    this.connect();
  }

  private connect() {
    const wsUrl =
      import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/live-feed";

    // Use native WebSocket since the backend uses native WebSockets
    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log("WebSocket connected");
        this.reconnectAttempts = 0;
        this.emit("connect", null);
      };

      ws.onclose = () => {
        console.log("WebSocket disconnected");
        this.handleReconnect();
      };

      ws.onerror = (error) => {
        console.error("WebSocket connection error:", error);
        this.handleReconnect();
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          if (message.type) {
            this.emit(message.type, message.data);
          }
        } catch (e) {
          console.error("Error parsing WebSocket message:", e);
        }
      };

      // Store the WebSocket instance
      this.socket = ws;
    } catch (error) {
      console.error("Failed to connect to WebSocket:", error);
      this.handleReconnect();
    }
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

  // This method is now used directly in the onmessage handler
  // private handleMessage(message: WebSocketMessage) {
  //   this.emit(message.type, message.data);
  // }

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
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      const message = JSON.stringify({ type: event, data });
      this.socket.send(message);
    } else {
      console.warn("WebSocket not connected, cannot send message");
    }
  }

  public reconnect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        // If already connected, just resolve
        if (this.socket?.readyState === WebSocket.OPEN) {
          resolve();
          return;
        }

        // Reset reconnect attempts
        this.reconnectAttempts = 0;

        // Connect and set up a one-time listener for connection success
        this.connect();

        if (this.socket) {
          const onConnect = () => {
            if (this.socket) {
              this.socket.removeEventListener("open", onConnect);
              resolve();
            }
          };

          const onError = (error: any) => {
            if (this.socket) {
              this.socket.removeEventListener("error", onError);
              reject(error);
            }
          };

          // Set timeout to avoid hanging forever
          const timeout = setTimeout(() => {
            if (this.socket) {
              this.socket.removeEventListener("open", onConnect);
              this.socket.removeEventListener("error", onError);
              reject(new Error("Connection timeout"));
            }
          }, 5000);

          this.socket.addEventListener("open", () => {
            clearTimeout(timeout);
            onConnect();
          });

          this.socket.addEventListener("error", (error) => {
            clearTimeout(timeout);
            onError(error);
          });
        } else {
          reject(new Error("Failed to initialize socket"));
        }
      } catch (error) {
        reject(error);
      }
    });
  }

  public disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    this.listeners.clear();
  }

  public isConnected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }
}

// Create singleton instance
const webSocketService = new WebSocketService();

export default webSocketService;
