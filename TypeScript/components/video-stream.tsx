"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import styles from "./video-stream.module.css"

interface VideoStreamProps {
  streamUrl: string
  title: string
  isActive?: boolean
}

export const VideoStream: React.FC<VideoStreamProps> = ({ streamUrl, title, isActive = true }) => {
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const attemptedRef = useRef<boolean>(false)

  // Reset state when stream URL changes or when toggled
  useEffect(() => {
    // Clear any existing retry timeout
    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current)
      retryTimeoutRef.current = null
    }

    // Reset loading state only if we haven't attempted to load this stream before
    if (!attemptedRef.current) {
      setIsLoading(true)
      setError(null)
    }

    if (imgRef.current) {
      imgRef.current.onload = () => {
        setIsLoading(false)
        setError(null)
        attemptedRef.current = true
      }

      imgRef.current.onerror = () => {
        setIsLoading(false)
        setError("Stream unavailable")
        attemptedRef.current = true

        // Set up automatic retry every 5 seconds
        retryTimeoutRef.current = setTimeout(() => {
          if (imgRef.current && isActive) {
            // Don't set loading to true on retry attempts to avoid the loading spinner
            imgRef.current.src = `${streamUrl}?t=${Date.now()}`
          }
        }, 5000)
      }
    }

    // Clean up timeout on unmount or when dependencies change
    return () => {
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current)
      }
    }
  }, [streamUrl, isActive])

  if (!isActive) {
    return null
  }

  return (
    <div className={styles.streamContainer}>
      <h2 className={styles.streamTitle}>{title}</h2>

      {isLoading && !attemptedRef.current && (
        <div className={styles.loadingContainer}>
          <div className={styles.loadingSpinner}></div>
          <p>Connecting to stream...</p>
        </div>
      )}

      <div className={styles.streamWrapper}>
        {error ? (
          <div className={styles.placeholderContainer}>
            <p className={styles.placeholderText}>
              {title} - {error}
            </p>
          </div>
        ) : (
          <img
            ref={imgRef}
            src={`${streamUrl}?t=${Date.now()}`}
            alt={`${title} stream`}
            className={styles.streamImage}
            style={{
              display: isLoading && !attemptedRef.current ? "none" : "block",
            }}
          />
        )}
      </div>
    </div>
  )
}
