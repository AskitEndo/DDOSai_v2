import React, { useEffect } from "react";
import { useAppContext } from "../context/AppContext";
import webSocketService from "../services/websocket";
import { Wifi, WifiOff } from "lucide-react";

const ConnectionStatus: React.FC = () => {
  const { state, dispatch } = useAppContext();

  useEffect(() => {
    // Check connection status initially
    const updateConnectionStatus = () => {
      dispatch({
        type: "SET_CONNECTION_STATUS",
        payload: webSocketService.isConnected(),
      });
    };

    // Update status immediately
    updateConnectionStatus();

    // Set up interval to check connection status
    const interval = setInterval(updateConnectionStatus, 5000);

    // Set up event listeners for connection changes
    const onConnect = () => {
      dispatch({ type: "SET_CONNECTION_STATUS", payload: true });
    };

    const onDisconnect = () => {
      dispatch({ type: "SET_CONNECTION_STATUS", payload: false });
    };

    // Add event listeners to the WebSocket service
    webSocketService.on("connect", onConnect);
    webSocketService.on("disconnect", onDisconnect);

    return () => {
      clearInterval(interval);
      webSocketService.off("connect", onConnect);
      webSocketService.off("disconnect", onDisconnect);
    };
  }, [dispatch]);

  return (
    <div className="flex items-center space-x-2">
      {state.isConnected ? (
        <>
          <Wifi className="w-4 h-4 text-success" />
          <span className="text-success text-sm">Connected</span>
        </>
      ) : (
        <>
          <WifiOff className="w-4 h-4 text-danger" />
          <span className="text-danger text-sm">Disconnected</span>
        </>
      )}
    </div>
  );
};

export default ConnectionStatus;
