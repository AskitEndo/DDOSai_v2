import React, { useEffect, useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navigation from "./components/Navigation";
import ErrorBoundary from "./components/ErrorBoundary";
import webSocketService from "./services/websocket";
import { useAppContext } from "./context/AppContext";
import {
  generateRandomDetections,
  generateRandomMetrics,
  generateRandomNetworkGraph,
} from "./utils/dummyData";
import { Database, AlertCircle } from "lucide-react";

// Use dynamic imports to fix TypeScript errors
const Dashboard = React.lazy(() => import("./components/Dashboard"));
const Simulation = React.lazy(() => import("./components/Simulation"));
const Logs = React.lazy(() => import("./components/Logs"));

// Component to load dummy data
const LoadDummyDataButton = ({ isOfflineMode = false }) => {
  const { dispatch } = useAppContext();
  const [isLoading, setIsLoading] = useState(false);

  const loadDummyData = () => {
    setIsLoading(true);

    // Generate random data
    const detections = generateRandomDetections(50);
    const metrics = generateRandomMetrics();
    const networkGraph = generateRandomNetworkGraph();

    // Update the app state with dummy data immediately to prevent flickering
    dispatch({ type: "SET_DETECTIONS", payload: detections });
    dispatch({ type: "SET_METRICS", payload: metrics });
    dispatch({ type: "SET_NETWORK_GRAPH", payload: networkGraph });
    dispatch({ type: "SET_CONNECTION_STATUS", payload: true });

    // Short delay just for the button loading state
    setTimeout(() => {
      setIsLoading(false);
    }, 300);
  };

  // Use different styles based on whether we're in offline mode
  const buttonClasses = isOfflineMode
    ? "ml-auto sm:ml-4 px-4 py-1.5 bg-blue-900/50 hover:bg-blue-800/70 text-blue-300 rounded-md flex items-center text-sm transition-colors"
    : "px-3 py-1 bg-blue-900/30 hover:bg-blue-800/50 text-blue-300/80 rounded-md flex items-center text-xs transition-colors";

  return (
    <button
      onClick={loadDummyData}
      disabled={isLoading}
      className={buttonClasses}
    >
      <Database className={isOfflineMode ? "w-4 h-4 mr-2" : "w-3 h-3 mr-1"} />
      {isLoading ? "Loading..." : "Load Sample Data"}
    </button>
  );
};

// Component to check backend availability
const CheckBackendButton = ({
  onCheck,
}: {
  onCheck: () => Promise<boolean>;
}) => {
  const [isChecking, setIsChecking] = useState(false);

  const checkBackend = async () => {
    setIsChecking(true);
    await onCheck();
    setIsChecking(false);
  };

  return (
    <button
      onClick={checkBackend}
      disabled={isChecking}
      className="ml-2 px-4 py-1.5 bg-gray-700/50 hover:bg-gray-600/70 text-gray-300 rounded-md flex items-center text-sm transition-colors"
    >
      <AlertCircle className="w-4 h-4 mr-2" />
      {isChecking ? "Checking..." : "Check Backend"}
    </button>
  );
};

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [connectionError, setConnectionError] = useState(false);

  const { dispatch } = useAppContext();

  // We're removing the unused loadDummyData function since it's now handled by the LoadDummyDataButton component

  useEffect(() => {
    // Initialize the app
    const initializeApp = async () => {
      try {
        // Set initial connection status to false until we confirm connection
        dispatch({ type: "SET_CONNECTION_STATUS", payload: false });

        // Try to connect to WebSocket
        try {
          await webSocketService.reconnect();
          // If connection successful, clear offline mode flag and set connection status to true
          localStorage.removeItem("ddosai_offline_mode");
          setConnectionError(false);
          dispatch({ type: "SET_CONNECTION_STATUS", payload: true });
        } catch (wsError) {
          console.warn(
            "WebSocket connection failed, continuing in offline mode"
          );
          setConnectionError(true);
          // Store offline status in localStorage to persist across page reloads
          localStorage.setItem("ddosai_offline_mode", "true");

          // Do NOT automatically load dummy data - wait for user to click the button
        }

        setIsLoading(false);
      } catch (error) {
        console.error("Error initializing app:", error);
        setConnectionError(true);
        setIsLoading(false);
      }
    };

    initializeApp();

    // Cleanup WebSocket on unmount
    return () => {
      try {
        webSocketService.disconnect();
      } catch (error) {
        console.warn("Error disconnecting WebSocket:", error);
      }
    };
  }, [dispatch]);

  // We're removing the automatic data loading effect
  // This will prevent the app from automatically loading sample data in offline mode

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 flex items-center justify-center">
        <div className="text-center p-8 bg-gray-800/50 rounded-xl border border-gray-700 shadow-2xl backdrop-blur-sm">
          <img
            src="/DDOSaiNOBG.png"
            alt="DDoS.AI Logo"
            className="w-24 h-24 object-contain mx-auto mb-6 animate-pulse-custom"
          />
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-xl font-medium text-blue-400">
            Loading DDoS.AI Platform...
          </p>
          <p className="text-gray-400 mt-2">Initializing security systems</p>
        </div>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <Router>
        <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-900 to-gray-800 text-white font-mono">
          <Navigation />

          {connectionError && (
            <div className="bg-red-900/30 border-b border-red-700 py-3 px-4">
              <div className="container mx-auto flex flex-col sm:flex-row sm:items-center">
                <div className="flex items-center mb-2 sm:mb-0">
                  <AlertCircle className="w-5 h-5 text-red-400 mr-2" />
                  <p className="text-red-300 text-sm">
                    Connection to backend services failed. Running in offline
                    mode.
                  </p>
                </div>
                <div className="flex items-center">
                  <LoadDummyDataButton isOfflineMode={true} />
                  <CheckBackendButton
                    onCheck={async () => {
                      try {
                        await webSocketService.reconnect();
                        setConnectionError(false);
                        dispatch({
                          type: "SET_CONNECTION_STATUS",
                          payload: true,
                        });
                        localStorage.removeItem("ddosai_offline_mode");
                        return true;
                      } catch (error) {
                        console.error("Failed to reconnect:", error);
                        return false;
                      }
                    }}
                  />
                </div>
              </div>
            </div>
          )}

          <main className="container mx-auto px-4 py-6">
            <React.Suspense
              fallback={
                <div className="flex flex-col items-center justify-center py-12">
                  <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-4"></div>
                  <span className="text-blue-400 font-medium">
                    Loading content...
                  </span>
                </div>
              }
            >
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/simulation" element={<Simulation />} />
                <Route path="/logs" element={<Logs />} />
              </Routes>
            </React.Suspense>
          </main>

          <footer className="border-t border-gray-800 py-4 mt-8">
            <div className="container mx-auto px-4">
              <div className="flex flex-col md:flex-row justify-between items-center">
                <div className="flex items-center mb-4 md:mb-0">
                  <img
                    src="/DDOSaiNOBG.png"
                    alt="DDoS.AI Logo"
                    className="w-6 h-6 mr-2"
                  />
                  <span className="text-gray-400 text-sm">
                    DDoS.AI Platform &copy; {new Date().getFullYear()}
                  </span>
                </div>
                <div className="flex space-x-6">
                  <a
                    href="#"
                    className="text-gray-400 hover:text-blue-400 text-sm"
                  >
                    Documentation
                  </a>
                  <a
                    href="#"
                    className="text-gray-400 hover:text-blue-400 text-sm"
                  >
                    API
                  </a>
                  <a
                    href="#"
                    className="text-gray-400 hover:text-blue-400 text-sm"
                  >
                    Support
                  </a>
                </div>
              </div>
            </div>
          </footer>
        </div>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
