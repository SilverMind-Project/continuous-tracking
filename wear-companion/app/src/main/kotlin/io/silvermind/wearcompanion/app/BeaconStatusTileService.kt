package io.silvermind.wearcompanion.app

import androidx.wear.protolayout.LayoutElementBuilders
import androidx.wear.protolayout.ResourceBuilders
import androidx.wear.protolayout.TimelineBuilders
import androidx.wear.tiles.RequestBuilders
import androidx.wear.tiles.TileBuilders
import androidx.wear.tiles.TileService
import com.google.common.util.concurrent.Futures
import com.google.common.util.concurrent.ListenableFuture
import io.silvermind.wearcompanion.core.TileStatus

/** Wear OS Tile displaying Advertising, Idle (not worn), or Bluetooth off. */
class BeaconStatusTileService : TileService() {
    override fun onTileRequest(requestParams: RequestBuilders.TileRequest): ListenableFuture<TileBuilders.Tile> {
        val statusText =
            when (CompanionForegroundService.lastTileStatus) {
                TileStatus.UNPAIRED -> getString(R.string.status_unpaired)
                TileStatus.ADVERTISING -> getString(R.string.status_advertising)
                TileStatus.IDLE_NOT_WORN -> getString(R.string.status_idle_not_worn)
                TileStatus.BLUETOOTH_OFF -> getString(R.string.status_bluetooth_off)
            }
        val root =
            LayoutElementBuilders.Box.Builder()
                .addContent(
                    LayoutElementBuilders.Text.Builder()
                        .setText(statusText)
                        .build(),
                )
                .build()
        val timeline =
            TimelineBuilders.Timeline.Builder()
                .addTimelineEntry(
                    TimelineBuilders.TimelineEntry.Builder()
                        .setLayout(LayoutElementBuilders.Layout.Builder().setRoot(root).build())
                        .build(),
                )
                .build()
        val tile =
            TileBuilders.Tile.Builder()
                .setResourcesVersion(RESOURCES_VERSION)
                .setTileTimeline(timeline)
                .build()
        return Futures.immediateFuture(tile)
    }

    override fun onTileResourcesRequest(
        requestParams: RequestBuilders.ResourcesRequest,
    ): ListenableFuture<ResourceBuilders.Resources> {
        val resources = ResourceBuilders.Resources.Builder().setVersion(RESOURCES_VERSION).build()
        return Futures.immediateFuture(resources)
    }

    companion object {
        private const val RESOURCES_VERSION = "1"
    }
}
