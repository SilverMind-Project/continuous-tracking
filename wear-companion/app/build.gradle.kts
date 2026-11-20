plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "io.silvermind.wearcompanion.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "io.silvermind.wearcompanion"
        minSdk = 34
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
    }

    flavorDimensions += "tier"
    productFlavors {
        create("bleOnly") {
            dimension = "tier"
            applicationIdSuffix = ".ble"
            buildConfigField("boolean", "TIER2_UWB_ENABLED", "false")
        }
        create("dualMode") {
            dimension = "tier"
            buildConfigField("boolean", "TIER2_UWB_ENABLED", "true")
        }
    }

    buildFeatures {
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_21
        targetCompatibility = JavaVersion.VERSION_21
    }

    kotlinOptions {
        jvmTarget = "21"
    }

    lint {
        warningsAsErrors = false
        abortOnError = true
    }
}

dependencies {
    implementation(project(":core"))
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.wear.tiles)
    implementation(libs.androidx.wear.protolayout)
    implementation(libs.androidx.wear.protolayout.material)
    implementation(libs.kotlinx.coroutines.core)

    "dualModeImplementation"(libs.androidx.uwb)
    "dualModeImplementation"(libs.play.services.nearby)
    "dualModeImplementation"(libs.eclipse.paho.mqtt)
}
