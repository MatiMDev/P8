"use client"

import type React from "react"

import { useState } from "react"
import { VideoStream } from "./video-stream"
import { VehicleDataPanel } from "./vehicle-data-panel"
import styles from "./stream-container.module.css"

// Define the stream types we might have
type StreamType = "rgb" | "depth" | "segmentation" | "thermal" | "detection" | "edges"

interface Stream {
  id: string
  type: StreamType
  title: string
  url: string
  active: boolean
}

export const StreamContainer: React.FC = () => {
  // Initial streams setup - all streams active by default
  const [streams, setStreams] = useState<Stream[]>([
    {
      id: "rgb",
      type: "rgb",
      title: "RGB Camera",
      url: "http://localhost:5000/video/rgb",
      active: true,
    },
    {
      id: "depth",
      type: "depth",
      title: "Depth Estimation",
      url: "http://localhost:5000/video/depth",
      active: true,
    },
    {
      id: "segmentation",
      type: "segmentation",
      title: "Segmentation",
      url: "http://localhost:5000/video/segmentation",
      active: true,
    },
    {
      id: "detection",
      type: "detection",
      title: "Object Detection",
      url: "http://localhost:5000/video/detection",
      active: true,
    },
    {
      id: "thermal",
      type: "thermal",
      title: "Thermal",
      url: "http://localhost:5000/video/thermal",
      active: true,
    },
    {
      id: "edges",
      type: "edges",
      title: "Edge Detection",
      url: "http://localhost:5000/video/edges",
      active: true,
    },
  ])

  // Function to toggle stream visibility
  const toggleStream = (id: string) => {
    setStreams(streams.map((stream) => (stream.id === id ? { ...stream, active: !stream.active } : stream)))
  }

  // Function to toggle all streams on/off
  const toggleAllStreams = (shouldBeActive: boolean) => {
    setStreams(streams.map((stream) => ({ ...stream, active: shouldBeActive })))
  }

  // Count active streams to determine grid layout
  const activeStreams = streams.filter((stream) => stream.active)
  const gridClass = getGridClass(activeStreams.length)

  return (
    <div className={styles.container}>
      <div className={styles.controls}>
        <div className={styles.controlsHeader}>
          <h2 className={styles.controlsTitle}>Camera Feeds</h2>
          <div className={styles.controlButtons}>
            <button className={styles.controlButton} onClick={() => toggleAllStreams(true)}>
              Show All
            </button>
            <button className={styles.controlButton} onClick={() => toggleAllStreams(false)}>
              Hide All
            </button>
          </div>
        </div>
        <div className={styles.toggles}>
          {streams.map((stream) => (
            <button
              key={stream.id}
              className={`${styles.toggleButton} ${stream.active ? styles.active : ""}`}
              onClick={() => toggleStream(stream.id)}
            >
              {stream.title}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.mainContent}>
        <div className={`${styles.streamsGrid} ${styles[gridClass]}`}>
          {activeStreams.map((stream) => (
            <VideoStream key={stream.id} streamUrl={stream.url} title={stream.title} isActive={stream.active} />
          ))}
        </div>

        <div className={styles.dataPanelContainer}>
          <VehicleDataPanel />
        </div>
      </div>
    </div>
  )
}

// Helper function to determine grid class based on number of active streams
function getGridClass(count: number): string {
  if (count === 1) return "gridSingle"
  if (count === 2) return "gridTwo"
  if (count <= 4) return "gridFour"
  if (count <= 6) return "gridSix"
  return "gridMany"
}
