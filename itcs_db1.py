import csv
from typing import List

import dash
import dash_cytoscape as cyto
import dash_html_components as html

authors: List[str] = ["Sznajd-Weron K", "Janutka A", "Bieniek M", "Bujkiewicz L", "Brzezińska M", "Gawarecki K",
                      "Gawełczyk M", "Hajdusianek A", "Herbrych J", "Jęzejewski A", "Karwat P", "Klonowski M",
                      "Kowalski P", "Kuchta M", "Lewandowski A", "Łydżba P", "Machnikowski P", "Major J", "Maśka M",
                      "Mielnik-Pyszczorski A", "Mierzejewski M", "Mituś A", "Morayne M", "Mulak M", "Mulak W",
                      "Mydlarczyk W", "Pawlik G", "Pawłowski J", "Radosz W", "Roszak K", "Sajna A", "Sitek A",
                      "Surówka P", "Trzmiel J", "Wigger D", "Wójs A", "Abramiuk-Szurlej A", "Brzuszek K", "Bugajny P",
                      "Groll D", "Hahn T", "Kawa K", "Krzykowski M", "Kupczyński M", "Kuśmierz B", "Nowak B",
                      "Rzepkowski B", "Środa M"
                      ]

prof = [
    "Sznajd-Weron K", "Klonowski M", "Machnikowski P", "Maśka M", "Mierzejewski M", "Mituś A", "Morayne M", "Wójs A"
]

dr = [
    "Janutka A", "Bieniek M", "Bujkiewicz L", "Brzezińska M", "Gawarecki K", "Gawełczyk M", "Hajdusianek A",
    "Herbrych J", "Jęzejewski A", "Karwat P", "Kowalski P", "Kuchta M", "Lewandowski A", "Łydżba P", "Major J",
    "Mielnik-Pyszczorski A", "Mulak M", "Mulak W", "Mydlarczyk W", "Pawlik G", "Pawłowski J", "Radosz W",
    "Roszak K", "Sajna A", "Sitek A", "Surówka P", "Trzmiel J", "Wigger D"
]

mgr = [
    "Abramiuk-Szurlej A", "Brzuszek K", "Bugajny P", "Kawa K", "Krzykowski M", "Kupczyński M", "Kuśmierz B",
    "Nowak B", "Rzepkowski B", "Środa M"
]

none = [
    "Groll D",
    "Hahn T",
]


def should_be_added(row_authors: List[str]) -> bool:
    for file_author in row_authors:
        for a in authors:
            if file_author in a:
                return True
    return False


def create_csv_database() -> None:
    with open('database.csv', mode='w') as result_file:
        result_file.write("Authors, Author(s) ID, Title, Year\n")
        with open('scopus.csv', newline='') as csvfile:
            input_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in input_reader:
                row_authors: List[str] = row[0].replace(",", "").split(". ")
                if row_authors[len(row_authors) - 1][-1] == ".":
                    row_authors[len(row_authors) - 1] = row_authors[len(row_authors) - 1][:-1]
                if should_be_added(row_authors):
                    content = "\"" + ",".join(row_authors) + "\"," + "\"" + row[1] + "\",\"" + row[2] + "\"," + \
                              row[3] + "\n"
                    result_file.write(content)


def create_graph(input_file_dir='database.csv') -> dict:
    authors_networks = {author: set([]) for author in authors}
    with open(input_file_dir, newline='') as csvfile:
        input_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(input_reader)
        for row in input_reader:
            pwr_authors = list(filter(lambda a: a in authors_networks, row[0].split(",")))
            for i in range(len(pwr_authors)):
                for j in range(len(pwr_authors)):
                    if i != j:
                        authors_networks[pwr_authors[j]].add(pwr_authors[i])

    return authors_networks


def create_network_graph():
    person_on_title = {}
    for pair in [('prof', prof), ('dr', dr), ('mgr', mgr), ('none', none)]:
        for person in pair[1]:
            person_on_title[person] = pair[0]

    app = dash.Dash(__name__)
    graph_data = []
    dict_data: dict = create_graph()
    for key in dict_data:
        graph_data.append({
            'data': {'id': key, 'label': key + "(" + str(len(dict_data[key])) + ")"},
            'classes': person_on_title[key]
        })
        values: set = dict_data[key]
        for v in values:
            graph_data.append({'data': {'source': key, 'target': v}})

    app.layout = html.Div([
        html.P("Collaboration between researchers, Author Albert Piekielny"),
        cyto.Cytoscape(
            id='cytoscape',
            elements=graph_data,
            layout={
                'name': 'concentric',
            },
            style={'width': '1200px', 'height': '900px'},
            stylesheet=[
                {
                    'selector': 'node',
                    'style': {
                        'width': 20,
                        'height': 20,
                        'label': 'data(label)',
                        "text-valign": "top",
                        "text-halign": "top",
                        'background-color': '#ff9d0a',
                        "font-weight": "bold",
                    }
                },
                {
                    'selector': '*',
                    'style': {
                        'line-color': 'grey',
                        'width': 2,
                        "opacity": 0.8,
                        "line-style": "dashed",
                    }
                },
                # Class selectors
                {
                    'selector': '.prof',
                    'style': {
                        'background-color': 'red',
                        'line-color': 'red',
                        "width": "70",
                        "height": "70",
                        "border-color": "#b51d1d",
                        "border-width": 3,
                        "border-opacity": 0.8,
                        "opacity": 0.8,
                    },
                },
                {
                    'selector': '.mgr',
                    'style': {
                        'background-color': '#007eff',
                        'line-color': '#007eff',
                        'border-color': '#04488e',
                        "border-width": 2.5,
                        "border-opacity": 0.8,
                        "opacity": 0.8,
                        "width": "45",
                        "height": "45",
                    },
                },
                {
                    'selector': '.dr',
                    'style': {
                        'background-color': 'green',
                        'line-color': 'green',
                        "width": "25",
                        "height": "25",
                        'border-color': '#316926',
                        "border-width": 2,
                        "border-opacity": 0.8,
                        "opacity": 0.8,
                    },
                },
                {
                    'selector': '.none',
                    'style': {
                        'background-color': 'orange',
                        'line-color': 'orange',
                        "width": "15",
                        "height": "15",
                        'border-color': '#a6761d',
                        "border-width": 1.5,
                        "border-opacity": 0.8,
                        "opacity": 0.8,
                    },
                },
            ]
        )
    ])

    app.run_server(debug=True)


if __name__ == '__main__':
    create_network_graph()


"""
java code

//        String regex = "Dr|dr|hab.|inż.|Mgr|PWr|[Pp]rof.|,";
//        String regex = "Dr|dr|hab.|inż.|Mgr|PWr|prof.|,";
//        String regex = "dr|hab.|inż.|Mgr|PWr|[Pp]rof.|,";
        String regex = "[Dd]r|hab.|inż.|PWr|[Pp]rof.|,";
        String names = \"""
            Prof. dr hab. Katarzyna Sznajd-Weron
            Dr hab. inż. Andrzej Janutka, prof. PWr
            Dr Maciej Bieniek
            Dr inż. Liliana Bujkiewicz
            Dr Marta Brzezińska
            Dr inż. Krzysztof Gawarecki
            Dr inż. Michał Gawełczyk
            Dr inż. Anna Hajdusianek, prof. PWr
            Dr Jacek Herbrych
            Dr Arkadiusz Jędrzejewski
            Dr inż. Paweł Karwat
            Prof. dr hab. inż. Marek Klonowski
            Dr Piotr Kowalski
            Dr Małgorzata Kuchta
            Dr Adrian Lewandowski
            Dr inż. Patrycja Łydżba
            Prof. dr hab. inż. Paweł Machnikowski
            Dr Jan Major
            Prof. dr hab. Maciej Maśka
            Dr Adam Mielnik-Pyszczorski
            Prof. dr hab. Marcin Mierzejewski
            Prof. dr hab. Antoni Mituś
            Prof. dr hab. Michał Morayne
            Dr Maciej Mulak
            Dr Wojciech Mulak
            Dr hab. Wojciech Mydlarczyk, prof. PWr
            Dr hab. inż. Grzegorz Pawlik, prof. PWr
            Dr inż. Jarosław Pawłowski
            Dr Wojciech Radosz
            Dr hab. inż. Katarzyna Roszak, prof. PWr
            Dr Adam Sajna
            Dr hab. inż. Anna Sitek, prof. PWr
            Dr Piotr Surówka
            Dr inż. Justyna Trzmiel
            Dr Daniel Wigger
            Prof. dr hab. inż. Arkadiusz Wójs
            Mgr inż. Angelika Abramiuk-Szurlej
            Mgr Kacper Brzuszek
            Mgr Paweł Bugajny
            Daniel Groll
            Thilo Hahn
            Mgr Karol Kawa
            Mgr Mateusz Krzykowski
            Mgr inż. Michał Kupczyński
            Mgr Bartosz Kuśmierz
            Mgr inż. Bartłomiej Nowak
            Mgr inż. Bartosz Rzepkowski
            Mgr inż. Maksymilian Środa\""";

        Arrays.stream(names.replaceAll(regex, "")
                .split("\n"))
                .map(String::trim)
//                .filter(x -> x.contains("Prof"))
                .filter(x -> x.contains("Mgr"))
                .map(x -> {
                    final String[] s = x.split(" +");
                    return s[s.length-1]+" "+s[s.length-2].charAt(0);
                })
                .forEach(x -> System.out.println("\"" + x + "\","));
"""
