import random
from datasets import load_dataset
from bench_datasets.base_dataset import BaseDataset

class ChatDoctorDataset(BaseDataset):
    def __init__(
        self,
        index_split: str = "train",
        test_split: str = "train",
        max_index_docs: int | None = 200,
        max_test_queries: int | None = 50,
        random_seed: int = 42,
    ):
        self.index_split = index_split
        self.test_split = test_split
        self.max_index_docs = max_index_docs
        self.max_test_queries = max_test_queries
        self.random_seed = random_seed

        self._documents: list[str] = []
        self._queries: list[dict] = []

    def load(self) -> None:
        dataset_name = "lavita/ChatDoctor-HealthCareMagic-100k"
        try:
            dataset = load_dataset(dataset_name, split="train")
        except Exception as e:
            raise RuntimeError(f"No se pudo cargar {dataset_name}. Error: {e}")
        
        all_data = list(dataset)
        rng = random.Random(self.random_seed)
        rng.shuffle(all_data)

        # 1. ÍNDICE: Concatenamos Input + Output
        index_limit = self.max_index_docs if self.max_index_docs else 200
        index_data = all_data[:index_limit]
        
        self._documents = []
        for row in index_data:
            # Obtenemos la pregunta (preferimos 'input', si no 'instruction')
            p_input = row.get("input", "").strip() or row.get("instruction", "").strip()
            p_output = row.get("output", "").strip()
            
            if p_input and p_output:
                # CREAMOS EL DOCUMENTO UNIFICADO
                # Usar etiquetas claras ayuda al modelo a distinguir roles
                full_doc = f"Patient: {p_input}\nDoctor's Advice: {p_output}"
                self._documents.append(full_doc)

        # 2. QUERIES: Se mantienen igual (buscamos por la duda del paciente)
        test_limit = self.max_test_queries if self.max_test_queries else 50
        test_data = all_data[index_limit : index_limit + test_limit]
        
        queries = []
        for row in test_data:
            patient_query = row.get("input", "").strip() or row.get("instruction", "").strip()
            answer = row.get("output", "").strip()
            
            if patient_query:
                queries.append({
                    "query": patient_query,
                    "answer": answer
                })
        
        self._queries = queries

    def get_documents(self) -> list[str]:
        return self._documents

    def get_queries(self) -> list[dict]:
        return self._queries

    def get_private_entities(self) -> list[str]:
        return ["disease", "medication", "symptoms"]