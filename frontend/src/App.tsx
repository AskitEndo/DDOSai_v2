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
        console.log("Initializing app and checking connection status...");
        // Set initial connection status to false until we confirm connection
        dispatch({ type: "SET_CONNECTION_STATUS", payload: false });

        // Add a timeout to ensure we don't get stuck in loading state
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => {
            reject(new Error("Connection timeout"));
          }, 3000); // 3 second timeout
        });

        // Try to connect to WebSocket with a timeout
        try {
          console.log("Attempting to connect to WebSocket...");
          await Promise.race([webSocketService.reconnect(), timeoutPromise]);
          // If connection successful, clear offline mode flag and set connection status to true
          console.log("WebSocket connection successful");
          localStorage.removeItem("ddosai_offline_mode");
          setConnectionError(false);
          dispatch({ type: "SET_CONNECTION_STATUS", payload: true });
        } catch (wsError) {
          console.warn(
            "WebSocket connection failed, continuing in offline mode:",
            wsError
          );
          setConnectionError(true);
          // Store offline status in localStorage to persist across page reloads
          localStorage.setItem("ddosai_offline_mode", "true");
          // Do NOT automatically load dummy data - wait for user to click the button
        }
      } catch (error) {
        console.error("Error initializing app:", error);
        setConnectionError(true);
      } finally {
        // Always set loading to false, regardless of connection status
        setIsLoading(false);
      }
    };

    // Start initialization with a safety timeout
    const safetyTimeout = setTimeout(() => {
      console.warn(
        "Safety timeout triggered - forcing app to load in offline mode"
      );
      setConnectionError(true);
      setIsLoading(false);
      localStorage.setItem("ddosai_offline_mode", "true");
      dispatch({ type: "SET_CONNECTION_STATUS", payload: false });

      // Show a message to the user
      console.log("Showing offline mode message to user");
    }, 5000); // 5 second safety timeout

    initializeApp().finally(() => {
      clearTimeout(safetyTimeout);
    });

    // Set up a periodic check for WebSocket connection
    const connectionCheckInterval = setInterval(() => {
      if (!webSocketService.isConnected()) {
        console.log("WebSocket not connected, attempting to reconnect...");
        // Don't wait for reconnect to complete
        webSocketService
          .reconnect()
          .then(() => {
            console.log("WebSocket reconnection successful");
            setConnectionError(false);
            dispatch({ type: "SET_CONNECTION_STATUS", payload: true });
            localStorage.removeItem("ddosai_offline_mode");
          })
          .catch((error) => {
            console.warn("WebSocket reconnection failed:", error);
          });
      }
    }, 30000); // Check every 30 seconds

    // Cleanup WebSocket and interval on unmount
    return () => {
      clearInterval(connectionCheckInterval);
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
                    mode. Some features like simulation require a backend
                    connection.
                  </p>
                </div>
                <div className="flex items-center">
                  <LoadDummyDataButton isOfflineMode={true} />
                  <CheckBackendButton
                    onCheck={async () => {
                      try {
                        console.log("Manually checking backend connection...");

                        // Try multiple endpoints to check if the backend is available
                        const endpoints = [
                          "http://localhost:8000/health",
                          "http://127.0.0.1:8000/health",
                        ];

                        let backendAvailable = false;

                        for (const endpoint of endpoints) {
                          try {
                            console.log(`Trying endpoint: ${endpoint}`);
                            const controller = new AbortController();
                            const timeoutId = setTimeout(
                              () => controller.abort(),
                              1000
                            );

                            const response = await fetch(endpoint, {
                              method: "GET",
                              headers: { "Content-Type": "application/json" },
                              signal: controller.signal,
                            });

                            clearTimeout(timeoutId);

                            if (response.ok) {
                              console.log(
                                `Backend health check successful at ${endpoint}!`
                              );
                              backendAvailable = true;
                              break;
                            }
                          } catch (fetchError) {
                            console.warn(
                              `Health check failed for ${endpoint}:`,
                              fetchError
                            );
                          }
                        }

                        if (!backendAvailable) {
                          console.warn(
                            "All health checks failed, trying WebSocket directly"
                          );
                        }

                        // Try to connect via WebSocket using the force connect method
                        const connected = await webSocketService.forceConnect();

                        if (connected) {
                          console.log("Backend connection successful!");
                          setConnectionError(false);
                          dispatch({
                            type: "SET_CONNECTION_STATUS",
                            payload: true,
                          });
                          localStorage.removeItem("ddosai_offline_mode");

                          // Force a reload of the page to ensure everything is properly connected
                          window.location.reload();
                        } else {
                          throw new Error("Failed to connect to backend");
                        }
                        return true;
                      } catch (error) {
                        console.error("Failed to reconnect:", error);

                        // Try one more time with a direct WebSocket connection
                        try {
                          console.log("Trying direct WebSocket connection...");
                          const ws = new WebSocket(
                            "ws://localhost:8000/ws/live-feed"
                          );

                          // Set a timeout to close the socket if it doesn't connect
                          const timeoutId = setTimeout(() => {
                            if (ws.readyState !== WebSocket.OPEN) {
                              ws.close();
                              alert(
                                "Failed to connect to backend. Please make sure the backend is running on http://localhost:8000"
                              );
                            }
                          }, 2000);

                          ws.onopen = () => {
                            clearTimeout(timeoutId);
                            console.log(
                              "Direct WebSocket connection successful!"
                            );
                            setConnectionError(false);
                            dispatch({
                              type: "SET_CONNECTION_STATUS",
                              payload: true,
                            });
                            localStorage.removeItem("ddosai_offline_mode");
                            window.location.reload();
                          };

                          ws.onerror = () => {
                            clearTimeout(timeoutId);
                            alert(
                              "Failed to connect to backend. Please make sure the backend is running on http://localhost:8000"
                            );
                          };
                        } catch (wsError) {
                          console.error(
                            "Direct WebSocket connection failed:",
                            wsError
                          );
                          alert(
                            "Failed to connect to backend. Please make sure the backend is running on http://localhost:8000"
                          );
                        }

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
