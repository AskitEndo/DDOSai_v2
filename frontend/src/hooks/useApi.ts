import { useState, useEffect, useCallback } from "react";
import { ApiResponse } from "../types";

interface UseApiOptions {
  immediate?: boolean;
  refreshInterval?: number;
}

/**
 * Custom hook for making API calls with loading state and refresh functionality
 * @param apiCall Function that returns a Promise with API response
 * @param options Configuration options
 */
export const useApi = <T>(
  apiCall: () => Promise<ApiResponse<T>>,
  options: UseApiOptions = {}
) => {
  const { immediate = false, refreshInterval = 0 } = options;

  const [data, setData] = useState<ApiResponse<T> | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);

  const execute = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiCall();
      setData(response);
      return response;
    } catch (err: any) {
      setError(err);
      return null;
    } finally {
      setLoading(false);
    }
  }, [apiCall]);

  // Initial fetch if immediate is true
  useEffect(() => {
    if (immediate) {
      execute();
    }
  }, [execute, immediate]);

  // Set up refresh interval if specified
  useEffect(() => {
    if (refreshInterval && refreshInterval > 0) {
      const intervalId = setInterval(() => {
        execute();
      }, refreshInterval);

      return () => clearInterval(intervalId);
    }
  }, [execute, refreshInterval]);

  return {
    data,
    loading,
    error,
    refresh: execute,
  };
};
