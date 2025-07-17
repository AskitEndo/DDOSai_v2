import React, { useEffect, useState, Suspense } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navigation from "./components/Navigation";
import ErrorBoundary from "./components/ErrorBoundary";
import webSocketService from "./services/websocket";

// Use dynamic imports to fix TypeScript errors
const Dashboard = React.lazy(() => import("./components/Dashboard"));
const Simulation = React.lazy(() => import("./components/Simulation"));
const Logs = React.lazy(() => import("./components/Logs"));

function App() {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Initialize the app
    const initializeApp = async () => {
      try {
        // Add a small delay to ensure everything is loaded
        await new Promise((resolve) => setTimeout(resolve, 100));
        setIsLoading(false);
      } catch (error) {
        console.error("Error initializing app:", error);
        setIsLoading(false);
      }
    };

    initializeApp();

    // Cleanup WebSocket on unmount
    return () => {
      webSocketService.disconnect();
    };
  }, []);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-white">Loading DDoS.AI Platform...</p>
        </div>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <Router>
        <div className="min-h-screen bg-gray-900 text-white font-mono">
          <Navigation />
          <main className="container mx-auto px-4 py-6">
            <React.Suspense
              fallback={
                <div className="flex items-center justify-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mr-3"></div>
                  <span className="text-white">Loading...</span>
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
        </div>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
