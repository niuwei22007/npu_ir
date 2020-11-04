adb remount
adb shell "rm -fr /data/log/android_logs/*"
adb shell "rm -fr /data/local/tmp/*"

rm -fr android_logs

adb push "libs/arm64-v8a/." "/data/local/tmp/"
adb push "data/." "/data/local/tmp/"

adb shell "cd /data/local/tmp; chmod +x test; export LD_LIBRARY_PATH=/data/local/tmp; ./test"
adb pull "/data/log/android_logs" "."