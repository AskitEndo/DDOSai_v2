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
    // Try different WebSocket URLs
    const urls = [
      import.meta.env.VITE_WS_URL,
      "ws://localhost:8000/ws/live-feed",
      "ws://127.0.0.1:8000/ws/live-feed",
      "ws://0.0.0.0:8000/ws/live-feed",
    ].filter(Boolean); // Remove any undefined/empty values

    const wsUrl = urls[0]; // Start with the first URL

    console.log("Attempting to connect to WebSocket at:", wsUrl);

    // Use native WebSocket since the backend uses native WebSockets
    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log("WebSocket connected successfully to:", wsUrl);
        this.reconnectAttempts = 0;
        this.emit("connect", null);

        // Force a dispatch to update connection status
        const event = new CustomEvent("websocket-connected");
        window.dispatchEvent(event);
      };

      ws.onclose = (event) => {
        console.log(
          "WebSocket disconnected with code:",
          event.code,
          "reason:",
          event.reason
        );

        // Try the next URL if this is the first attempt
        if (this.reconnectAttempts === 0 && urls.length > 1) {
          console.log("Trying next WebSocket URL");
          // Rotate the URLs array to try the next one
          urls.push(urls.shift()!);
        }

        this.handleReconnect();
      };

      ws.onerror = (error) => {
        console.error("WebSocket connection error:", error);
        // Don't handle reconnect here, let onclose handle it
      };

      ws.onmessage = (event) => {
        try {
          console.log("WebSocket message received:", event.data);
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
          console.log("WebSocket already connected, no need to reconnect");
          resolve();
          return;
        }

        console.log("Attempting to reconnect WebSocket...");

        // Reset reconnect attempts
        this.reconnectAttempts = 0;

        // Close any existing socket
        if (this.socket) {
          try {
            this.socket.close();
          } catch (e) {
            // Ignore errors when closing
          }
          this.socket = null;
        }

        // Try to connect WebSocket
        this.connect();

        if (this.socket) {
          const onConnect = () => {
            if (this.socket) {
              this.socket.removeEventListener("open", onConnect);
              console.log("WebSocket reconnected successfully");
              resolve();
            }
          };

          const onError = (error: any) => {
            if (this.socket) {
              this.socket.removeEventListener("error", onError);
              console.error("WebSocket reconnection error:", error);
              reject(error);
            }
          };

          // Set timeout to avoid hanging forever
          const timeout = setTimeout(() => {
            if (this.socket) {
              this.socket.removeEventListener("open", onConnect);
              this.socket.removeEventListener("error", onError);
              console.error("WebSocket connection timeout");
              reject(new Error("Connection timeout"));
            }
          }, 3000); // 3 second timeout

          this.socket.addEventListener("open", () => {
            clearTimeout(timeout);
            onConnect();
          });

          this.socket.addEventListener("error", (error) => {
            clearTimeout(timeout);
            onError(error);
          });
        } else {
          console.error("Failed to initialize socket");
          reject(new Error("Failed to initialize socket"));
        }
      } catch (error) {
        console.error("WebSocket reconnection error:", error);
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

  public forceConnect(): Promise<boolean> {
    return new Promise((resolve) => {
      // Try all possible WebSocket URLs
      const urls = [
        "ws://localhost:8000/ws/live-feed",
        "ws://127.0.0.1:8000/ws/live-feed",
        "ws://0.0.0.0:8000/ws/live-feed",
      ];

      let currentIndex = 0;
      let connected = false;

      const tryNextUrl = () => {
        if (currentIndex >= urls.length || connected) {
          resolve(connected);
          return;
        }

        const url = urls[currentIndex++];
        console.log(`Trying to connect to WebSocket at: ${url}`);

        try {
          const ws = new WebSocket(url);

          ws.onopen = () => {
            console.log(`WebSocket connected successfully to: ${url}`);
            this.socket = ws;
            connected = true;
            this.emit("connect", null);
            resolve(true);
          };

          ws.onerror = () => {
            console.log(`Failed to connect to: ${url}`);
            ws.close();
            setTimeout(tryNextUrl, 500);
          };

          ws.onclose = () => {
            if (!connected) {
              setTimeout(tryNextUrl, 500);
            }
          };

          // Set a timeout for this connection attempt
          setTimeout(() => {
            if (ws.readyState !== WebSocket.OPEN) {
              ws.close();
              if (!connected) {
                setTimeout(tryNextUrl, 500);
              }
            }
          }, 1000);
        } catch (error) {
          console.error(`Error connecting to ${url}:`, error);
          setTimeout(tryNextUrl, 500);
        }
      };

      // Start trying URLs
      tryNextUrl();
    });
  }
}

// Create singleton instance
const webSocketService = new WebSocketService();

export default webSocketService;
