import { NextResponse } from "next/server"

export async function GET() {
  try {
    // Fetch data from the Python server
    const response = await fetch("http://localhost:5000/data/vehicle", {
      headers: {
        Accept: "application/json",
        "Cache-Control": "no-cache",
      },
      // This will be executed on the server side, avoiding CORS issues
      next: { revalidate: 0 }, // Don't cache the response
    })

    if (!response.ok) {
      throw new Error(`Failed to fetch data: ${response.status}`)
    }

    const data = await response.json()

    // Return the data as JSON
    return NextResponse.json(data)
  } catch (error) {
    console.error("Error fetching vehicle data:", error)
    return NextResponse.json({ error: "Failed to fetch vehicle data" }, { status: 500 })
  }
}
