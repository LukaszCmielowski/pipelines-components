from kfp import dsl

from autorag.rag_templates_optimization.src.steps.chunking import chunk_documents
from autorag.rag_templates_optimization.src.steps.indexing import index
from ai4rag.core.experiment.experiment import AI4RAGExperiment
from autorag.rag_templates_optimization.src.utils import set_search_space_defaults
from pathlib import Path
import tarfile

@dsl.component(
    # base_image="python:3.13",
    target_image="rag_base:test"
)
def rag_templates_optimization(
    extracted_text: dsl.InputPath(str),
    test_data: dsl.InputPath(str),
    validated_configurations: dsl.InputPath(dict),
    leaderboard: dsl.Output[dsl.Artifact],  #? 
    rag_patterns: dsl.Output[dsl.Artifact]
    vector_database_id: str | None = None,
    optimization_settings: dict | None = None,
    # metrics? 
):
    """
    Rag Templates Optimization component.

    A very detailed description for the component.

    Args:
        TBD
    
    Returns:
        leaderboard: an Artifact containing ranked RAG Patterns.
        rag_patterns: an Artifact containing best RAG Patterns. 

    """

    docs_folder = Path(extracted_text)
    assert docs_folder.is_dir()


    set_search_space_defaults()

    chunked_docs = chunk_documents()

    # setup vector store
    index()

    #retrieve documents && run optimisation loop
    rag_exp = AI4RAGExperiment.search()

    rag_patterns_dir = Path("$HOME/rag_exp/rag_patterns").mkdir(parents=True)

    for patt in rag_exp.evaluations:
        patt_dir = rag_patterns_dir / patt.name
        (patt_dir / "pattern.json").write(patt)
        (patt_dir / "inference_notebook.ipynb").write(patt.inference_code)
        (patt_dir / "indexing_notebook.ipynb").write(patt.indexing_code)
    
    with tarfile.open(rag_patterns.path, "w") as archive:
        archive.add(rag_patterns_dir)


    

if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        rag_templates_optimization,
        package_path=__file__.replace(".py", "_component.yaml"),
    )


# locally running components and pipelines, implement for debuggin at least!
