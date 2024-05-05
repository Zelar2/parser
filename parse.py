import json
from typing import Iterable
import logging
import httpx
import asyncio
import functools

# Define constants
URL_GET_PROJECTS_INFO = "https://xn--80ajpld2c.xn--80af5akm8c.xn--p1ai/award/api/v1/projects/get-projects-info"
URL_BASE_ITEM = "https://xn--80ajpld2c.xn--80af5akm8c.xn--p1ai/award/api/v1/applications/"
OUTPUT_FILE_NAME = "output_25595.json"

sem = asyncio.Semaphore(10)

headers = {
    'Cookie': 'k8s=94d25a75c593bada99b95711693b133a|1615a4e152e65e9370f04d2c0d48c2a0',
    'User-Agent': 'PostmanRuntime/7.37.3',
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive'
}

json_conf = {
    "take": 25595,
    "skip": 0,
    "activeTab": 1,
    "searchText": "",
    "hasAdvancedFeatures": False,
    "isColleague": False,
    "currentVotingId": "f978781b-b2b0-4b52-838f-8ec8a5c7ea5b"
}

logging.basicConfig(level=logging.INFO, filename="parse.log", filemode="w")


async def supervisor(project_ids: Iterable[str]):
    """
    Define supervisor task
    :param project_ids: string with project ids
    """
    logging.info('Starting supervisor...')

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=10), timeout=60) as client:
        client.headers.update(headers)

        tasks = [parse_project(client, i, id_) for i, id_ in enumerate(project_ids)]
        tasks_iter: Iterable[dict | None] = await asyncio.gather(*tasks)
        await asyncio.to_thread(write_json, tasks_iter)

    logging.info('Ending supervisor...')


def write_json(data: Iterable[dict]):
    """
    Write json data to file
    :param data: dict to write
    """
    with open(OUTPUT_FILE_NAME, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))


def handle_errors():
    """
    Handle exceptions
    """

    def wrapper(func):
        """
        Decorator function
        :param func: input function
        :return: wrapped function or raise exception
        """

        @functools.wraps(func)
        async def wrapped(*args, **kwargs):
            """
            Wrap function
            :param args: arguments
            :param kwargs: keyword arguments
            :return: wrapped function
            """
            try:
                return await func(*args, **kwargs)
            except httpx.HTTPStatusError as exc:
                print(f'HTTP error {exc.response.status_code} - {exc.response.reason_phrase} on URL {exc.request.url}')
            except httpx.RequestError as exc:
                print(f'{exc} {type(exc)}'.strip())

        return wrapped

    return wrapper


def get_projects(url=URL_GET_PROJECTS_INFO):
    """
    Get all projects
    :param url: base url
    :return: list of projects
    """
    with httpx.Client(timeout=200) as client:
        client.headers.update(headers)

        r = client.post(url, json=json_conf)
        data = r.json()
        projects = data['projects']
        public_ids = [p['projectPublicId'] for p in projects]

        return public_ids


@handle_errors()
async def parse_project(client: httpx.AsyncClient, i: int, project_id: str = None, url: str = URL_BASE_ITEM) -> dict:
    """
    Get project info
    :param client: client to get project info
    :param i: iteration
    :param project_id: project id
    :param url: base api url
    :return: dict with projects info
    """
    result = {}
    item_columns = ['applicationName', 'shortDescriptionRaw', 'socialGroupsRaw', 'aimsRaw',
                    'tasksRaw', 'socialDescriptionRaw', 'startDate', 'finalDate']
    result_columns = ['qualityResult', 'evaluation']

    async with sem:
        response = await client.get(url + f'/item?applicationId={project_id}')
        response.raise_for_status()

    data = response.json()
    contest = data.get('contestDirectionTenant')

    result['id'] = i
    result['url'] = f'https://оценка.гранты.рф/award/project/{project_id}'
    if not contest:
        result['category'] = 'None'
    else:
        result['category'] = contest.get('name')
    for item in item_columns:
        result[item] = data.get(item)

    async with sem:
        response = await client.get(url + f'/results?applicationId={project_id}')
        response.raise_for_status()

    data = response.json()

    for item in result_columns:
        result[item] = data.get(item)

    print(f'HTTP code 200 for project: {i}')
    return result


def main():
    """
    Main function
    """
    project_ids = get_projects()
    asyncio.run(supervisor(project_ids))


if __name__ == '__main__':
    main()
