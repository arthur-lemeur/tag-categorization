import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    """Fixture pour créer un client de test FastAPI"""
    return TestClient(app)

def test_api_route(client):
    """Test de la route GET /api"""
    response = client.get('/api')
    assert response.status_code == 200
    # Pour FastAPI, utilisez response.json() au lieu de response.json
    assert response.json() == {"message": "Hello World!"}

def test_predict_tags_route():
    with TestClient(app) as client:
        response = client.post("/predict", json={
            "Title": "use fastapi",
            "Body": "need help api rout",
            "Merged_doc": "use fastapi need help api rout"
        })
        assert response.status_code == 200
    
    response_data = response.json()
    # Vérifie que la clé existe
    assert "predicted_labels" in response_data
    assert isinstance(response_data["predicted_labels"], list)

# Test pour les cas d'erreur
def test_predict_tags_invalid_input(client):
    """Test avec des données invalides"""
    response = client.post('/predict', json={
        "title": "",  # titre vide
        "body": ""    # corps vide
    })
    # Adaptez le code de statut selon votre implémentation
    # (peut être 400, 422, ou 200 selon votre logique)
    assert response.status_code in [200, 400, 422]

def test_predict_tags_missing_fields(client):
    """Test avec des champs manquants"""
    response = client.post('/predict', json={
        "title": "Test title"
        # 'body' manquant
    })
    # FastAPI retourne généralement 422 pour les champs manquants
    assert response.status_code == 422