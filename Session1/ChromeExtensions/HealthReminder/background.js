const DEFAULT_REMINDER_INTERVAL = 30; // minutes
let keepAliveInterval;

function createAlarm(interval) {
  chrome.alarms.create("healthReminder", {
    periodInMinutes: interval
  });
  console.log(`Alarm created with interval: ${interval} minutes`);
}

function getCountdown(callback) {
  chrome.alarms.get("healthReminder", (alarm) => {
    if (alarm) {
      const remainingTime = Math.max(0, Math.floor((alarm.scheduledTime - Date.now()) / 1000));
      const minutes = Math.floor(remainingTime / 60);
      const seconds = remainingTime % 60;
      callback(`${minutes}m ${seconds}s`);
    } else {
      console.log("No alarm found, creating a new one");
      chrome.storage.local.get(['reminderInterval'], (result) => {
        const interval = result.reminderInterval || DEFAULT_REMINDER_INTERVAL;
        createAlarm(interval);
        callback(`${interval}m 0s`);
      });
    }
  });
}

function showNotification() {
  chrome.notifications.create({
    type: "basic",
    iconUrl: "icon128.png",
    title: "Health Reminder",
    message: "Time to close your eyes for 20 seconds and take a sip of water!",
    priority: 2
  });
}

function keepAlive() {
  if (keepAliveInterval) clearInterval(keepAliveInterval);
  keepAliveInterval = setInterval(() => {
    console.log("Keeping service worker alive");
    chrome.runtime.getPlatformInfo(() => {});
  }, 20000); // Every 20 seconds
}

chrome.runtime.onInstalled.addListener(() => {
  console.log("Extension installed");
  chrome.storage.local.get(['reminderInterval'], (result) => {
    const interval = result.reminderInterval || DEFAULT_REMINDER_INTERVAL;
    createAlarm(interval);
  });
  keepAlive();
});

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === "healthReminder") {
    console.log("Alarm triggered");
    showNotification();
    // Recreate the alarm for the next interval
    chrome.storage.local.get(['reminderInterval'], (result) => {
      const interval = result.reminderInterval || DEFAULT_REMINDER_INTERVAL;
      createAlarm(interval);
    });
  }
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === 'local' && changes.reminderInterval) {
    console.log(`Reminder interval changed to ${changes.reminderInterval.newValue} minutes`);
    createAlarm(changes.reminderInterval.newValue);
  }
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "getCountdown") {
    getCountdown(sendResponse);
    return true; // Indicates that the response is sent asynchronously
  }
});

// Keep the service worker alive
keepAlive();

// Restart the keep-alive interval when the service worker wakes up
chrome.runtime.onStartup.addListener(keepAlive);
