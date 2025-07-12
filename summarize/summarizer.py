from abc import ABC, abstractmethod

class Summarizer(ABC):
    @abstractmethod
    def summarize(self, abstract: str) -> str:
        """
        Summarizes the given abstract.
        :param abstract: The abstract text to summarize.
        :return: A summarized version of the abstract.
        """
        pass

    @abstractmethod
    def load_papers(self, path):
        """
        Loads papers from the specified path.
        :param path: The path to the papers file.
        :return: A list of papers.
        """
        pass

    @abstractmethod
    def save_papers(self, papers, path):
        """
        Saves the given papers to the specified path.
        :param papers: The list of papers to save.
        :param path: The path to the output file.
        """
        pass

    @abstractmethod
    def summarize_all(self, input_path: str, output_path: str):
        """
        Summarizes all papers from the input path and saves the results to the output path.
        :param input_path: The path to the input papers file.
        :param output_path: The path to the output file.
        """
        pass
