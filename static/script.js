const form = document.getElementById('predictionForm');
const result = document.getElementById('result');
const table = document.getElementById('table_body');
const loader = document.getElementById('loader');

async function loadModelInfo() {
    try {
        const response = await fetch('http://localhost:8000/model/infos', {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' },
        });

        const data = await response.json();

        if (!response.ok) {
            return;
        }
        const title = document.getElementById('model_infos_title');
        const id = document.getElementById('model_infos_id');
        title.textContent = data.model_type;
        id.textContent = "Run ID:" + data.run_id;

        const metricsContainer = document.getElementById('metrics_container');
        metricsContainer.innerHTML = '';
        if (data.performance && Object.keys(data.performance).length > 0) {
            const barCharts = createHorizontalBarCharts(data.performance);
            metricsContainer.appendChild(barCharts);
        } else {
            metricsContainer.textContent = 'Aucune métrique disponible';
        }

    } catch (err) {
        console.error(err);
    }
}

// Charger les infos au chargement de la page
loadModelInfo();

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const title = document.getElementById('title').value;
    const body = document.getElementById('body').value;

    loader.classList.remove('hidden');

    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ Title: title, Body: body })
        });

        const data = await response.json();

        if (!response.ok) {
            result.textContent = "Erreur : " + JSON.stringify(data, null, 2);
            return;
        }
        console.log(data)

        const tags_predicted = data.predicted_labels;
        const tags_probabilities = data.probability;

        result.innerHTML = ''; // Clear previous result
        table.innerHTML = '';  // Clear previous table rows

        if (Array.isArray(tags_predicted) && tags_predicted.length > 0) {
            const ul = document.createElement('ul');
            ul.classList.add('tag-list');

            tags_predicted.forEach((tag, i) => {
                // Affichage sous forme de liste
                const li = document.createElement('li');
                li.textContent = tag;
                li.classList.add('tag-item');
                ul.appendChild(li);

                // Affichage dans le tableau
                const tr = document.createElement('tr');
                const th_label = document.createElement('th');
                const td_proba = document.createElement('td');
                th_label.textContent = tag;
                td_proba.textContent = tags_probabilities && tags_probabilities[i] != null
                    ? `${(tags_probabilities[i] * 100).toFixed(2)} %`
                    : 'N/A';
                tr.appendChild(th_label);
                tr.appendChild(td_proba);
                table.appendChild(tr);
            });

            result.appendChild(ul);
        } else {
            result.textContent = "Aucune étiquette prédite.";
        }

    } catch (err) {
        result.textContent = "Erreur de connexion ou de traitement.";
        console.error(err);
    } finally {
        loader.classList.add('hidden');
    }
});

    function createHorizontalBarCharts(metricsData) {
    const container = document.createElement('div');
    
    // Titre du graphique
    const title = document.createElement('div');
    title.className = 'chart-title';
    title.textContent = 'Performances du Modèle';
    container.appendChild(title);
    
    // Statistiques résumées
    const values = Object.values(metricsData);
    const avgValue = values.reduce((a, b) => a + b, 0) / values.length;
    const maxValue = Math.max(...values);
    const minValue = Math.min(...values);
    
    // Créer les barres pour chaque métrique
    const chartContainer = document.createElement('div');
    chartContainer.className = 'chart-container';
    
    Object.entries(metricsData).forEach(([metric, value]) => {
        const metricBar = document.createElement('div');
        metricBar.className = 'metric-bar';
        
        // Label de la métrique
        const label = document.createElement('div');
        label.className = 'metric-label';
        label.textContent = metric;
        
        // Container de la barre
        const barContainer = document.createElement('div');
        barContainer.className = 'bar-container';
        
        // Barre de remplissage
        const barFill = document.createElement('div');
        barFill.className = `bar-fill ${metric}`;
        barFill.style.width = '0%'; // Animation depuis 0
        
        // Valeur dans la barre
        const barValue = document.createElement('div');
        barValue.className = 'bar-value';
        barValue.textContent = value.toFixed(3);
        
        // Valeur à droite
        const metricValue = document.createElement('div');
        metricValue.className = 'metric-value';
        metricValue.textContent = (value * 100).toFixed(1) + '%';
        
        // Assemblage
        barFill.appendChild(barValue);
        barContainer.appendChild(barFill);
        metricBar.appendChild(label);
        metricBar.appendChild(barContainer);
        metricBar.appendChild(metricValue);
        chartContainer.appendChild(metricBar);
        
        // Animation de la barre (après un court délai)
        setTimeout(() => {
            barFill.style.width = (value * 100) + '%';
        }, 100);
    });
    
    container.appendChild(chartContainer);
    return container;
}