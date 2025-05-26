"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import styles from "./vehicle-data-panel.module.css"

interface VehicleData {
  timestamp?: {
    groundtruth?: number
    imu?: number
  }
  groundtruth?: {
    position?: number[]
    rotation?: number[]
    velocity?: number[]
    angular_velocity?: number[]
    timestamp?: number
  }
  imu?: {
    acceleration?: number[]
    angular_velocity?: number[]
    orientation?: number[]
    timestamp?: number
  }
}

export const VehicleDataPanel: React.FC = () => {
  const [vehicleData, setVehicleData] = useState<VehicleData>({})
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [debugMode, setDebugMode] = useState(false)

  // Store raw response for debugging
  const [rawResponse, setRawResponse] = useState<string>("")

  // Reference to track if component is mounted
  const isMounted = useRef(true)

  useEffect(() => {
    // Set up cleanup function
    return () => {
      isMounted.current = false
    }
  }, [])

  useEffect(() => {
    const fetchVehicleData = async () => {
      try {
        // Use our Next.js API route instead of directly calling the Flask server
        // This avoids CORS issues since the request is made server-side
        const response = await fetch(`/api/vehicle-data?t=${Date.now()}`)

        // Store the raw response text for debugging
        const responseText = await response.text()
        setRawResponse(responseText)

        if (!response.ok) {
          throw new Error(`Server returned ${response.status}: ${responseText}`)
        }

        // Parse the JSON manually
        let data
        try {
          data = JSON.parse(responseText)
          if (isMounted.current) {
            setVehicleData(data)
            setIsLoading(false)
            setError(null)
            setLastUpdated(new Date())
          }
        } catch (parseError) {
          throw new Error(`Failed to parse JSON: ${responseText}`)
        }
      } catch (err) {
        if (isMounted.current) {
          setError(`${err instanceof Error ? err.message : "Unknown error"}`)
          setIsLoading(false)
        }
      }
    }

    // Initial fetch
    fetchVehicleData()

    // Set up polling interval
    const intervalId = setInterval(fetchVehicleData, 1000) // Update every second

    // Clean up on unmount
    return () => clearInterval(intervalId)
  }, [])

  // Format vector for display
  const formatVector = (vector: number[] | undefined) => {
    if (!vector || !Array.isArray(vector)) return "N/A"
    return vector.map((v) => v.toFixed(2)).join(", ")
  }

  // Format timestamp for display
  const formatTimestamp = (timestamp: number | undefined) => {
    if (!timestamp) return "N/A"
    return timestamp.toFixed(3) + "s"
  }

  // Toggle debug mode
  const toggleDebugMode = () => {
    setDebugMode(!debugMode)
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h2 className={styles.title}>Vehicle Data</h2>
        <button className={styles.debugButton} onClick={toggleDebugMode} title="Toggle debug mode">
          {debugMode ? "Hide Debug" : "Debug"}
        </button>
      </div>

      {isLoading && !Object.keys(vehicleData).length ? (
        <div className={styles.loadingState}>
          <div className={styles.spinner}></div>
          <p>Loading vehicle data...</p>
        </div>
      ) : error ? (
        <div className={styles.errorState}>
          <p className={styles.errorTitle}>Connection Error</p>
          <p className={styles.errorMessage}>{error}</p>
          <div className={styles.troubleshooting}>
            <p className={styles.troubleshootingTitle}>Troubleshooting:</p>
            <ol className={styles.troubleshootingList}>
              <li>Ensure the Python server is running on port 5000</li>
              <li>Check if the server is accessible from your Next.js app</li>
              <li>Verify the endpoint URL is correct</li>
              <li>Check server logs for errors</li>
            </ol>
          </div>
        </div>
      ) : (
        <>
          <div className={styles.dataContent}>
            <div className={styles.section}>
              <h3 className={styles.sectionTitle}>Timestamps</h3>
              <div className={styles.dataRow}>
                <span className={styles.dataLabel}>Groundtruth:</span>
                <span className={styles.dataValue}>{formatTimestamp(vehicleData.timestamp?.groundtruth)}</span>
              </div>
              <div className={styles.dataRow}>
                <span className={styles.dataLabel}>IMU:</span>
                <span className={styles.dataValue}>{formatTimestamp(vehicleData.timestamp?.imu)}</span>
              </div>
              {lastUpdated && (
                <div className={styles.dataRow}>
                  <span className={styles.dataLabel}>Last Update:</span>
                  <span className={styles.dataValue}>{lastUpdated.toLocaleTimeString()}</span>
                </div>
              )}
            </div>

            <div className={styles.section}>
              <h3 className={styles.sectionTitle}>Groundtruth</h3>
              <div className={styles.dataRow}>
                <span className={styles.dataLabel}>Position:</span>
                <span className={styles.dataValue}>{formatVector(vehicleData.groundtruth?.position)}</span>
              </div>
              <div className={styles.dataRow}>
                <span className={styles.dataLabel}>Rotation:</span>
                <span className={styles.dataValue}>{formatVector(vehicleData.groundtruth?.rotation)}</span>
              </div>
              <div className={styles.dataRow}>
                <span className={styles.dataLabel}>Velocity:</span>
                <span className={styles.dataValue}>{formatVector(vehicleData.groundtruth?.velocity)}</span>
              </div>
              <div className={styles.dataRow}>
                <span className={styles.dataLabel}>Angular Vel:</span>
                <span className={styles.dataValue}>{formatVector(vehicleData.groundtruth?.angular_velocity)}</span>
              </div>
            </div>

            <div className={styles.section}>
              <h3 className={styles.sectionTitle}>IMU Data</h3>
              <div className={styles.dataRow}>
                <span className={styles.dataLabel}>Acceleration:</span>
                <span className={styles.dataValue}>{formatVector(vehicleData.imu?.acceleration)}</span>
              </div>
              <div className={styles.dataRow}>
                <span className={styles.dataLabel}>Angular Vel:</span>
                <span className={styles.dataValue}>{formatVector(vehicleData.imu?.angular_velocity)}</span>
              </div>
              <div className={styles.dataRow}>
                <span className={styles.dataLabel}>Orientation:</span>
                <span className={styles.dataValue}>{formatVector(vehicleData.imu?.orientation)}</span>
              </div>
            </div>
          </div>

          {debugMode && (
            <div className={styles.debugSection}>
              <h3 className={styles.debugTitle}>Debug Information</h3>
              <div className={styles.debugContent}>
                <h4>Raw Response:</h4>
                <pre className={styles.rawResponse}>{rawResponse || "No data received"}</pre>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
