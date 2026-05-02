package com.edgeveda.edge_veda

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log

/**
 * Foreground service for sustained on-device inference on Android 12+.
 *
 * Background:
 * Android 12 (API 31) tightened the rules around long-running work:
 * if your app expects to run >30 seconds with the UI hidden, you
 * need to declare a foreground service (or live in JobScheduler /
 * WorkManager). RAG bulk-embed, long Whisper transcripts, and
 * extended chat sessions all routinely exceed 30s.
 *
 * This service exists purely so the OS won't kill the inference
 * worker when the user backgrounds the app mid-job. It does no
 * inference itself — the actual work runs on the existing
 * background isolate. The service is just a "please keep this
 * process alive" anchor with a notification.
 *
 * Lifecycle:
 * - Host app calls `startInferenceForegroundService(title, text)`
 *   before kicking off a long job.
 * - Host app calls `stopInferenceForegroundService()` when the job
 *   completes (success or failure). The host owns failure-mode
 *   cleanup; we don't auto-stop on errors here because the host
 *   may want to retry without a flicker.
 * - The notification is non-interactive — tapping it brings the
 *   app to the foreground (Android default behaviour for service
 *   notifications without contentIntent).
 *
 * Permissions:
 * - FOREGROUND_SERVICE — declared in manifest, granted at install.
 * - FOREGROUND_SERVICE_DATA_SYNC — Android 14+ requirement, granted
 *   at install.
 * - POST_NOTIFICATIONS — runtime permission on Android 13+. If the
 *   user denies it, the service still runs (the inference is
 *   protected) but the notification doesn't render. Host app
 *   should request POST_NOTIFICATIONS once before the first long
 *   job; for now we don't request it from inside the service
 *   because the host typically already has a UX moment to ask.
 */
class InferenceForegroundService : Service() {

    companion object {
        private const val TAG = "EdgeVedaFGService"
        private const val NOTIFICATION_ID = 4242
        private const val CHANNEL_ID = "edge_veda_inference"
        private const val CHANNEL_NAME = "On-device AI"
        private const val CHANNEL_DESC =
            "Keeps long inference jobs alive while the app is hidden"
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        ensureChannel()
        val title = intent?.getStringExtra("title") ?: "Edge Veda inference"
        val text = intent?.getStringExtra("text") ?: "Running on-device AI…"
        val notif = buildNotification(title, text)
        try {
            // Android 14+ requires the service-type to match the
            // foregroundServiceType in the manifest. dataSync is the
            // closest fit for long-running on-device computation
            // with no special hardware tie-in.
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                startForeground(
                    NOTIFICATION_ID,
                    notif,
                    android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC,
                )
            } else {
                startForeground(NOTIFICATION_ID, notif)
            }
        } catch (e: Exception) {
            Log.w(TAG, "startForeground failed: ${e.message}")
        }
        // START_NOT_STICKY: if the system kills us due to memory,
        // don't recreate. The host app re-issues the start command
        // when it kicks off the next job.
        return START_NOT_STICKY
    }

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (nm.getNotificationChannel(CHANNEL_ID) != null) return
        val channel = NotificationChannel(
            CHANNEL_ID,
            CHANNEL_NAME,
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = CHANNEL_DESC
            setShowBadge(false)
            // Don't make sound — this is housekeeping, not an alert.
            setSound(null, null)
        }
        nm.createNotificationChannel(channel)
    }

    private fun buildNotification(title: String, text: String): Notification {
        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
        }
        return builder
            .setContentTitle(title)
            .setContentText(text)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
    }
}
