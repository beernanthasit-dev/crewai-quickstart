import sys
from unittest.mock import MagicMock, patch

# Mock classes for crewai
class MockAgent:
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class MockTask:
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class MockCrew:
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class MockLLM:
    def __init__(self, *args, **kwargs):
        pass
    def call(self, *args, **kwargs):
        pass

# Mock crewai and its submodules before importing the module under test
mock_crewai = MagicMock()
mock_crewai.Agent = MockAgent
mock_crewai.Task = MockTask
mock_crewai.Crew = MockCrew
mock_crewai.LLM = MockLLM
mock_crewai.Process = MagicMock()

mock_crewai_project = MagicMock()

def mock_decorator(f):
    return f

mock_crewai_project.CrewBase = lambda x: x
mock_crewai_project.agent = mock_decorator
mock_crewai_project.task = mock_decorator
mock_crewai_project.crew = mock_decorator

sys.modules['crewai'] = mock_crewai
sys.modules['crewai.project'] = mock_crewai_project
sys.modules['crewai.agents.agent_builder.base_agent'] = MagicMock()

# Import the classes after mocking
from github_resume_generator.crew import GithubResumeGenerator, GeminiWithGoogleSearch

def test_gemini_with_google_search_call_no_tools():
    llm = GeminiWithGoogleSearch(model='test-model')
    messages = [{'role': 'user', 'content': 'test message'}]

    with patch.object(MockLLM, 'call') as mock_super_call:
        llm.call(messages)

        # Verify that googleSearch was inserted into tools
        args, kwargs = mock_super_call.call_args
        assert 'tools' in kwargs
        assert {'googleSearch': {}} in kwargs['tools']
        assert len(kwargs['tools']) == 1

def test_gemini_with_google_search_call_with_tools():
    llm = GeminiWithGoogleSearch(model='test-model')
    messages = [{'role': 'user', 'content': 'test message'}]
    existing_tools = [{'otherTool': {}}]

    with patch.object(MockLLM, 'call') as mock_super_call:
        llm.call(messages, tools=existing_tools)

        # Verify that googleSearch was inserted at the beginning
        args, kwargs = mock_super_call.call_args
        assert 'tools' in kwargs
        assert kwargs['tools'][0] == {'googleSearch': {}}
        assert kwargs['tools'][1] == {'otherTool': {}}

def test_github_resume_generator_agents():
    generator = GithubResumeGenerator()
    generator.agents_config = {
        'github_profile_researcher': {'role': 'GitHub Profile Analyst', 'goal': 'Analyze profile'},
        'resume_writer': {'role': 'Technical Resume Generator', 'goal': 'Generate resume'}
    }

    agent1 = generator.github_profile_researcher()
    assert agent1.config == generator.agents_config['github_profile_researcher']
    assert isinstance(agent1.llm, GeminiWithGoogleSearch)

    agent2 = generator.resume_writer()
    assert agent2.config == generator.agents_config['resume_writer']
    assert agent2.llm == 'gemini/gemini-2.5-flash-preview-04-17'

def test_github_resume_generator_tasks():
    generator = GithubResumeGenerator()
    generator.tasks_config = {
        'profile_research_task': {'description': 'Conduct research'},
        'resume_generation_task': {'description': 'Generate resume'}
    }

    task1 = generator.profile_research_task()
    assert task1.config == generator.tasks_config['profile_research_task']

    task2 = generator.resume_generation_task()
    assert task2.config == generator.tasks_config['resume_generation_task']

def test_github_resume_generator_crew():
    generator = GithubResumeGenerator()
    agents = [MagicMock(spec=MockAgent)]
    tasks = [MagicMock(spec=MockTask)]
    generator.agents = agents
    generator.tasks = tasks

    crew = generator.crew()
    assert isinstance(crew, MockCrew)
    assert crew.agents == agents
    assert crew.tasks == tasks
