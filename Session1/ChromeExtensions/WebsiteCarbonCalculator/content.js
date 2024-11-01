chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "pageLoaded") {
    calculateEmissions(request.url);
    sendResponse({received: true}); // Acknowledge receipt of the message
  }
});

function calculateEmissions(url) {
  chrome.runtime.sendMessage({ action: "calculateEmissions", url: url }, response => {
    if (response && response.success && response.data) {
      const data = response.data;
      const result = {
        co2: data.statistics && data.statistics.co2 && data.statistics.co2.grid ? 
             data.statistics.co2.grid.grams.toFixed(2) : 'N/A',
        cleanerThan: data.cleanerThan !== undefined ? 
                     (data.cleanerThan * 100).toFixed(2) : 'N/A',
        green: data.green === true,
        rating: data.rating || 'N/A'
      };
      chrome.storage.local.set({lastResult: result}, () => {
        chrome.runtime.sendMessage({ action: "updatePopup", result: result });
      });
    } else {
      console.error('Error calculating emissions:', response ? response.error : 'Invalid API response');
      const errorResult = {
        error: true,
        message: 'Unable to calculate emissions. The API response was invalid or incomplete.'
      };
      chrome.storage.local.set({lastResult: errorResult}, () => {
        chrome.runtime.sendMessage({ action: "updatePopup", result: errorResult });
      });
    }
  });
}

// Trigger calculation on initial page load
calculateEmissions(window.location.href);
