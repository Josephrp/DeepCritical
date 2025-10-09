"""
HYBRID PATENT INTELLIGENCE SYSTEM
- Conexión EPO (OPS API) + Selenium Google Patents
- Carga de credenciales desde .env
"""

import os
import requests
import base64
import json
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import re
from dotenv import load_dotenv

# === Cargar variables desde .env ===
load_dotenv()
EPO_CONSUMER_KEY = os.getenv("EPO_CONSUMER_KEY")
EPO_CONSUMER_SECRET = os.getenv("EPO_CONSUMER_SECRET")
BASE_URL = "https://ops.epo.org/3.2"


class HybridPatentSearcher:
    def __init__(self):
        """Inicializar buscador híbrido EPO + Selenium"""
        self.consumer_key = EPO_CONSUMER_KEY
        self.consumer_secret = EPO_CONSUMER_SECRET
        self.base_url = BASE_URL
        self.access_token = None
        self.epo_available = False

        # Selenium setup
        self.driver = None
        self.selenium_available = False

        self.setup_apis()

    def setup_apis(self):
        """Configurar ambas APIs"""
        print("🔧 Configurando APIs...")

        # Setup EPO
        try:
            self.authenticate_epo()
            self.epo_available = True
            print("✅ EPO API disponible")
        except Exception as e:
            print(f"⚠️ EPO no disponible: {e}")

        # Setup Selenium
        try:
            self.setup_selenium()
            self.selenium_available = True
            print("✅ Selenium disponible")
        except Exception as e:
            print(f"⚠️ Selenium no disponible: {e}")

        if not self.epo_available and not self.selenium_available:
            raise Exception("❌ Ninguna API está disponible")

    def authenticate_epo(self):
        """Autenticar con EPO"""
        auth_url = f"{self.base_url}/auth/accesstoken"
        credentials = f"{self.consumer_key}:{self.consumer_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()

        headers = {
            'Authorization': f'Basic {encoded}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        response = requests.post(auth_url, headers=headers, data='grant_type=client_credentials')
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data['access_token']
        else:
            raise Exception(f"EPO auth failed: {response.status_code}")

    def setup_selenium(self):
        """Configurar Selenium"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        self.driver = webdriver.Chrome(options=chrome_options)

    def search_epo(self, query, max_results=10):
        """Buscar en EPO (términos simples)"""
        if not self.epo_available:
            return []

        search_url = f"{self.base_url}/rest-services/published-data/search"
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        }
        params = {
            'q': query,
            'Range': f'1-{max_results}'
        }

        try:
            response = requests.get(search_url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                return self.process_epo_results(data, query)
            else:
                return []
        except Exception:
            return []

    def process_epo_results(self, data, query):
        """Procesar resultados de EPO"""
        try:
            world_data = data.get('ops:world-patent-data', {})
            biblio_search = world_data.get('ops:biblio-search', {})
            search_result = biblio_search.get('ops:search-result', {})
            publications = search_result.get('ops:publication-reference', [])

            if isinstance(publications, dict):
                publications = [publications]

            processed = []
            for pub in publications:
                doc_ids = pub.get('document-id', [])
                if isinstance(doc_ids, dict):
                    doc_ids = [doc_ids]

                for doc in doc_ids:
                    if doc.get('@document-id-type') == 'epodoc':
                        country = doc.get('country', {}).get('$', '')
                        number = doc.get('doc-number', {}).get('$', '')
                        kind = doc.get('kind', {}).get('$', '')
                        date = doc.get('date', {}).get('$', '')

                        processed.append({
                            'patent_id': f"{country}{number}{kind}",
                            'title': f"EPO Patent {country}{number}",
                            'country': country,
                            'date': date,
                            'source': 'EPO',
                            'query': query,
                            'link': f"https://worldwide.espacenet.com/patent/search/family/000000000/publication/{country}{number}?q={number}"
                        })
                        break

            return processed

        except Exception:
            return []

    def search_selenium(self, query, max_results=5):
        """Buscar con Selenium en Google Patents"""
        if not self.selenium_available:
            return []

        search_url = f"https://patents.google.com/?q={query.replace(' ', '%20')}"

        try:
            self.driver.get(search_url)
            time.sleep(3)

            elements = self.driver.find_elements(By.CSS_SELECTOR, "search-result-item")

            results = []
            for i, element in enumerate(elements[:max_results]):
                try:
                    element_text = element.text

                    # Buscar ID de patente
                    patent_id = None
                    id_patterns = [r'(US\d{7,}[AB]?\d?)', r'(EP\d{7,}[AB]?\d?)', r'(CN\d{8,}[AB]?)']
                    for pattern in id_patterns:
                        match = re.search(pattern, element_text)
                        if match:
                            patent_id = match.group(1)
                            break

                    lines = element_text.split('\n')
                    title = lines[0] if lines else "Google Patents Result"

                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', element_text)
                    date = date_match.group(1) if date_match else "Unknown"

                    link = None
                    try:
                        link_elem = element.find_element(By.CSS_SELECTOR, 'a')
                        link = link_elem.get_attribute('href')
                    except:
                        pass

                    results.append({
                        'patent_id': patent_id or f"GP{i+1}",
                        'title': title[:100],
                        'country': patent_id[:2] if patent_id else 'Unknown',
                        'date': date,
                        'source': 'Google Patents',
                        'query': query,
                        'link': link or search_url
                    })

                except Exception:
                    continue

            return results

        except Exception:
            return []

    def smart_search(self, query, max_results=10):
        """Búsqueda inteligente que combina ambas fuentes"""
        print(f"🔍 Búsqueda híbrida: '{query}'")

        results = []

        if self.epo_available:
            epo_results = self.search_epo(query, max_results // 2)
            results.extend(epo_results)
            if epo_results:
                print(f"   ✅ EPO: {len(epo_results)} patentes")
            else:
                print(f"   ❌ EPO: sin resultados")

        if self.selenium_available and len(results) < max_results:
            remaining = max_results - len(results)
            selenium_results = self.search_selenium(query, remaining)
            results.extend(selenium_results)
            if selenium_results:
                print(f"   ✅ Selenium: {len(selenium_results)} patentes")
            else:
                print(f"   ❌ Selenium: sin resultados")

        return results

    def close(self):
        """Cerrar recursos"""
        if self.driver:
            self.driver.quit()
            print("🚪 Selenium cerrado")


def main():
    print("🚀 HYBRID PATENT INTELLIGENCE SYSTEM")
    print("🔧 EPO + Google Patents")
    print("=" * 60)

    searcher = None
    try:
        searcher = HybridPatentSearcher()
        results = searcher.smart_search("plastic recycling", max_results=10)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if searcher:
            searcher.close()


if __name__ == "__main__":
    main()
