pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "wear-uwb-probe"

include(":core")

// Include the Wear OS Android :app module whenever an Android SDK is configured
// (ANDROID_HOME, ANDROID_SDK_ROOT, or local.properties sdk.dir) and not skipped.
val skipAndroid = providers.gradleProperty("skipAndroid").map { it.toBoolean() }.getOrElse(false)
val localProps = file("local.properties")
val hasLocalSdkDir = localProps.isFile && localProps.readText().contains("sdk.dir=")
val hasAndroidSdk =
    !System.getenv("ANDROID_HOME").isNullOrBlank() ||
        !System.getenv("ANDROID_SDK_ROOT").isNullOrBlank() ||
        hasLocalSdkDir

if (!skipAndroid && hasAndroidSdk) {
    include(":app")
}
