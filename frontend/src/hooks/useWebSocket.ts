import { useEffect } from "react";
import webSocketService from "../services/websocket";

/**
 * Custom hook for subscribing to WebSocket events
 * @param event The event name to subscribe to
 * @param callback The callback function to execute when the event is received
 */
export const useWebSocket = (event: string, callback: (data: any) => void) => {
  useEffect(() => {
    // Subscribe to the event
    const unsubscribe = webSocketService.on(event, callback);

    // Return cleanup function
    return () => {
      unsubscribe();
    };
  }, [event, callback]);
};
