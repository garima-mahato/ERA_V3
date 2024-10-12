const DEFAULT_REMINDER_INTERVAL = 30; // minutes

function updateCountdownUI() {
  chrome.runtime.sendMessage({ action: "getCountdown" }, (response) => {
    if (chrome.runtime.lastError) {
      console.error(chrome.runtime.lastError);
      return;
    }
    const countdownElement = document.getElementById("countdown");
    countdownElement.textContent = response;
  });
}

function resetTimer() {
  chrome.storage.local.get(['reminderInterval'], (result) => {
    const interval = result.reminderInterval || DEFAULT_REMINDER_INTERVAL;
    chrome.alarms.clear("healthReminder", () => {
      chrome.alarms.create("healthReminder", { delayInMinutes: interval });
      updateCountdownUI();
    });
  });
}

function setCustomInterval() {
  const intervalInput = document.getElementById("intervalInput");
  const newInterval = parseInt(intervalInput.value, 10);
  if (newInterval > 0) {
    chrome.storage.local.set({ reminderInterval: newInterval }, () => {
      chrome.alarms.clear("healthReminder", () => {
        chrome.alarms.create("healthReminder", { delayInMinutes: newInterval });
        updateCountdownUI();
      });
    });
  }
}

function initializePopup() {
  const resetTimerButton = document.getElementById("resetTimer");
  const setIntervalButton = document.getElementById("setInterval");
  const intervalInput = document.getElementById("intervalInput");

  if (resetTimerButton) {
    resetTimerButton.addEventListener("click", resetTimer);
  } else {
    console.error("Reset Timer button not found");
  }

  if (setIntervalButton) {
    setIntervalButton.addEventListener("click", setCustomInterval);
  } else {
    console.error("Set Interval button not found");
  }

  if (intervalInput) {
    chrome.storage.local.get(['reminderInterval'], (result) => {
      const interval = result.reminderInterval || DEFAULT_REMINDER_INTERVAL;
      intervalInput.value = interval;
    });
  } else {
    console.error("Interval input not found");
  }

  updateCountdownUI();
  setInterval(updateCountdownUI, 1000);
}

// Run initialization when the popup is opened
document.addEventListener('DOMContentLoaded', initializePopup);
