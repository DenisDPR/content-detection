# Aggressive Content Detection Library
### To use this library on your android project
### Step 1: Add the JitPack repository to your build file
Add it in your root build.gradle at the end of repositories
```
  allprojects {
	repositories {
		...
		maven { url 'https://jitpack.io' }
	}
  }
```
### Step 2: Add the dependency
```
  dependencies {
	implementation 'com.github.shebogholo:content-detection:0.0.2'
  }
```
### Step 3: To use prediction method
It accepts two arguments: (Context, text)
```
   DetectContent.detect(MainActivity.this, "मुझे उस से नफरत है");
```
