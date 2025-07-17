/**
 * API Error Handler Utility
 * Provides functions to handle API connectivity issues and display appropriate messages
 */

// Store the API connectivity status
let isApiConnected = true;
let lastCheckTime = 0;
const CHECK_INTERVAL = 10000; // 10 seconds

/**
 * Check if the API is available
 * @returns {Promise<boolean>} True if API is available, false otherwise
 */
export const checkApiAvailability = async () => {
  try {
    // Only check if the last check was more than CHECK_INTERVAL ms ago
    const now = Date.now();
    if (now - lastCheckTime < CHECK_INTERVAL) {
      return isApiConnected;
    }

    lastCheckTime = now;

    // Try to fetch the health endpoint with a short timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);

    const response = await fetch("/api/health", {
      method: "GET",
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (response.ok) {
      if (!isApiConnected) {
        console.log("API connection restored");
        // If we were previously disconnected, reload the app to restore functionality
        if (!isApiConnected) {
          window.location.reload();
        }
      }
      isApiConnected = true;
      return true;
    } else {
      isApiConnected = false;
      return false;
    }
  } catch (error) {
    console.error("API connectivity check failed:", error);
    isApiConnected = false;
    return false;
  }
};

/**
 * Handle API errors and display appropriate messages
 * @param {Error} error - The error object
 * @param {Function} setError - Function to set error state in the component
 */
export const handleApiError = (error, setError) => {
  if (error.name === "AbortError" || error.message === "Failed to fetch") {
    setError({
      title: "Connection Error",
      message:
        "Unable to connect to the backend service. Please check your connection or try again later.",
      isConnectionError: true,
    });

    // Start checking for API availability
    startApiAvailabilityCheck();
  } else {
    setError({
      title: "Error",
      message: error.message || "An unexpected error occurred",
      isConnectionError: false,
    });
  }
};

let apiCheckInterval = null;

/**
 * Start checking for API availability periodically
 */
export const startApiAvailabilityCheck = () => {
  if (!apiCheckInterval) {
    apiCheckInterval = setInterval(async () => {
      const isAvailable = await checkApiAvailability();
      if (isAvailable && !isApiConnected) {
        // API is back online, reload the app
        window.location.reload();
      }
    }, CHECK_INTERVAL);
  }
};

/**
 * Stop checking for API availability
 */
export const stopApiAvailabilityCheck = () => {
  if (apiCheckInterval) {
    clearInterval(apiCheckInterval);
    apiCheckInterval = null;
  }
};

/**
 * Get the current API connectivity status
 * @returns {boolean} True if API is connected, false otherwise
 */
export const getApiConnectivityStatus = () => {
  return isApiConnected;
};
