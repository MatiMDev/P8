import { StreamContainer } from "@/components/stream-container"
import styles from "./page.module.css"

export default function Home() {
  return (
    <main className={styles.main}>
      <header className={styles.header}>
        <h1 className={styles.title}>Unity Stream Viewer</h1>
        <p className={styles.description}>Real-time video streams from Unity via Python</p>
      </header>

      <StreamContainer />
    </main>
  )
}
