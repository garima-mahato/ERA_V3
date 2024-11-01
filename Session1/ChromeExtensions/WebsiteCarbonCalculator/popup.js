let urlSpan, urlContainer, ratingDiv, cleanerThanP, lastTestedSpan, co2P, energyP, yearlyStatsDiv;
let lastResult = null;

async function calculateEmissions(url) {
    clearResults();
    try {
        const response = await fetch(`https://api.websitecarbon.com/site?url=${encodeURIComponent(url)}`);
        const data = await response.json();
        console.log("API response:", data);
        const result = {
            url: url,
            co2: data.statistics?.co2?.grid?.grams ?? 'N/A',
            cleanerThan: data.cleanerThan !== undefined ? data.cleanerThan * 100 : 'N/A',
            green: data.green === true,
            rating: data.rating || 'N/A',
            energy: data.statistics?.energy ?? 'N/A'
        };
        console.log("Processed result:", result);
        await displayResult(result);
        await chrome.storage.local.set({lastResult: result});
    } catch (error) {
        console.error('Error:', error);
        const errorResult = {
            error: true,
            message: 'Unable to calculate emissions. An error occurred.'
        };
        await displayResult(errorResult);
        await chrome.storage.local.set({lastResult: errorResult});
    }
}

async function displayResult(result) {
    console.log("displayResult called with:", result);
    if (result.error) {
        ratingDiv.textContent = 'Error';
        cleanerThanP.textContent = result.message;
        clearOtherFields();
    } else {
        lastResult = result;
        if (result.url) {
            urlSpan.textContent = result.url;
            urlContainer.style.display = 'block';
            hasData = true;
        }
        ratingDiv.textContent = result.rating || 'N/A';
        ratingDiv.style.color = getRatingColor(result.rating);
        cleanerThanP.textContent = `This is ${result.cleanerThan !== 'N/A' ? 'cleaner' : 'dirtier'} than ${result.cleanerThan.toFixed(2)}% of all web pages globally`;
        lastTestedSpan.textContent = new Date().toLocaleDateString();
        co2P.textContent = `${result.co2 !== 'N/A' ? result.co2.toFixed(2) + 'g' : 'N/A'} of CO2 is produced every time someone visits this web page.`;
        energyP.textContent = `This web page ${result.green ? 'uses' : 'does not use'} green energy.`;
        await displayYearlyStats(result);
    }
}

async function displayYearlyStats(result) {
    console.log("displayYearlyStats called with:", result);

    const monthlyViews = 10000;
    const stat_values = {
        grams: parseFloat(result.co2) || 0,
        litres: (parseFloat(result.co2) || 0) * 0.001,
        energy: parseFloat(result.energy) || 0,
        monthly_views: monthlyViews
    };

    console.log("Parsed stat_values:", stat_values);

    const r = (value, decimals) => {
        const rounded = Number(Math.round(value + 'e' + decimals) + 'e-' + decimals);
        console.log(`Rounding ${value} to ${decimals} decimals: ${rounded}`);
        return isNaN(rounded) ? 0 : rounded;
    };

    const i = {
        monthly_views: () => monthlyViews,
        grams_per_year: () => stat_values.grams * stat_values.monthly_views * 12,
        litres_per_year: () => stat_values.litres * stat_values.monthly_views * 12,
        energy_per_year: () => stat_values.energy * stat_values.monthly_views * 12,
        co2_grams: () => r(stat_values.grams, 2),
        co2_kg: () => r(i.grams_per_year() / 1e3, 2),
        sumos: () => r(i.grams_per_year() / 15e4, 2),
        cupsOfTea: () => r(i.grams_per_year() / 7.38, 0),
        bubbles: () => {
            const t = r(1e6 * i.litres_per_year() / 0.52, 0);
            console.log(`Bubbles calculation: ${t}`);
            if (t === 0) return '0';
            return t >= 1e12 ? r(t / 1e12, 2) + " trillion" 
                 : t >= 1e9 ? r(t / 1e9, 2) + " billion"
                 : t >= 1e6 ? r(t / 1e6, 2) + " million"
                 : t >= 1e3 ? r(t / 1e3, 2) + " thousand"
                 : t.toString();
        },
        trees: () => Math.max(1, Math.ceil(i.co2_kg() / 22)),
        balloons: () => r(i.grams_per_year() / 14, 0),
        energy: () => r(i.energy_per_year(), 2),
        kilometres: () => {
            const result = r(6.4 * i.energy_per_year(), 0);
            console.log(`Kilometres calculation: ${result}`);
            return result;
        },
        smartphoneCharges: () => {
            const result = r(i.energy_per_year() / 0.012, 0);
            console.log(`Smartphone charges calculation: ${result}`);
            return result;
        }
    };

    yearlyStatsDiv.innerHTML = `
        <div class="section report-stats section--large section--background">
            <h2>Over a year, with ${i.monthly_views().toLocaleString()} monthly page views, 
              <div id="urlContainer" class="url-container"> 
              <span id="url2" class="url-text">${result.url}</span>
              </div>
            produces:</h2>
            <div class="stat-grid">
                <div class="stat-item">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8h1a4 4 0 0 1 0 8h-1"></path><path d="M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8z"></path><line x1="6" y1="1" x2="6" y2="4"></line><line x1="10" y1="1" x2="10" y2="4"></line><line x1="14" y1="1" x2="14" y2="4"></line></svg>
                    <span class="stat-value">${i.co2_kg()}</span>
                    <span class="stat-label">kg of CO2 equivalent</span>
                    <span class="stat-description">As much CO2 as boiling water for ${i.cupsOfTea()} cups of tea</span>
                </div>
                <div class="stat-item">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="2" width="14" height="20" rx="2" ry="2"></rect><line x1="12" y1="18" x2="12.01" y2="18"></line></svg>
                    <span class="stat-value">${i.smartphoneCharges()}</span>
                    <span class="stat-label">smartphone charges</span>
                    <span class="stat-description">As much CO2 as this many full charges of an average smartphone</span>
                </div>
                <div class="stat-item">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="5.5" cy="17.5" r="3.5"/>
                        <circle cx="12" cy="13" r="2"/>
                        <circle cx="19" cy="17.5" r="3.5"/>
                        <circle cx="7.5" cy="7.5" r="3.5"/>
                        <circle cx="17" cy="7.5" r="3.5"/>
                    </svg>
                    <span class="stat-value">${i.bubbles()}</span>
                    <span class="stat-label">bubbles</span>
                    <span class="stat-description">Woah, that's a lot of bubbles!</span>
                </div>
                <div class="stat-item">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 22v-5"/>
                        <path d="M9 17h6"/>
                        <path d="M12 3L3 16h18L12 3z"/>
                    </svg>
                    <span class="stat-value">${i.trees()}</span>
                    <span class="stat-label">trees</span>
                    <span class="stat-description">This web page emits the amount of carbon that ${i.trees()} trees absorb in a year.</span>
                </div>
                <div class="stat-item">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18.92 6.01C18.72 5.42 18.16 5 17.5 5h-11c-.66 0-1.21.42-1.42 1.01L3 12v8c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-1h12v1c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-8l-2.08-5.99zM6.5 16c-.83 0-1.5-.67-1.5-1.5S5.67 13 6.5 13s1.5.67 1.5 1.5S7.33 16 6.5 16zm11 0c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zM5 11l1.5-4.5h11L19 11H5z"></path></svg>
                    <span class="stat-value">${i.kilometres()}</span>
                    <span class="stat-label">km in an electric car</span>
                    <span class="stat-description">That's how far an electric car could drive with ${i.energy()} kWh of energy.</span>
                </div>
            </div>
        </div>
    `;
}

function clearResults() {
    urlSpan.textContent = '';
    ratingDiv.textContent = 'Calculating...';
    ratingDiv.style.color = '';
    cleanerThanP.textContent = '';
    lastTestedSpan.textContent = '';
    co2P.textContent = '';
    energyP.textContent = '';
    yearlyStatsDiv.innerHTML = '';
}

function clearOtherFields() {
    urlSpan.textContent = '';
    lastTestedSpan.textContent = '';
    co2P.textContent = '';
    energyP.textContent = '';
    yearlyStatsDiv.innerHTML = '';
}

function getRatingColor(rating) {
    const colors = {
        'A+': '#0cce6b',
        'A': '#0cce6b',
        'B': '#97cc64',
        'C': '#ffc83f',
        'D': '#ff9a36',
        'E': '#ff602c',
        'F': '#ff2c2c'
    };
    return colors[rating] || '#000000';
}

document.addEventListener('DOMContentLoaded', async function() {
    const calculateButton = document.getElementById('calculate');
    urlSpan = document.getElementById('url');
    urlContainer = document.getElementById('urlContainer');
    ratingDiv = document.getElementById('rating');
    cleanerThanP = document.getElementById('cleanerThan');
    lastTestedSpan = document.getElementById('lastTested');
    co2P = document.getElementById('co2');
    energyP = document.getElementById('energy');
    yearlyStatsDiv = document.getElementById('yearlyStats');

    calculateButton.addEventListener('click', async function() {
        const tabs = await chrome.tabs.query({active: true, currentWindow: true});
        const currentUrl = tabs[0].url;
        await calculateEmissions(currentUrl);
    });

    // Display the last calculated result when popup opens
    const result = await chrome.storage.local.get(['lastResult']);
    if (result.lastResult) {
        await displayResult(result.lastResult);
    } else {
        ratingDiv.textContent = 'No data';
        cleanerThanP.textContent = 'No calculation results yet. Please click "Test again".';
    }

    // Calculate emissions for the current tab when popup opens
    const tabs = await chrome.tabs.query({active: true, currentWindow: true});
    const currentUrl = tabs[0].url;
    await calculateEmissions(currentUrl);
});
