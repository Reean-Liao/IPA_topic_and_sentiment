import pandas as pd
import xml.etree.ElementTree as xml


def read_xml(path: str) -> pd.DataFrame:
    """
    Args:
        path (str): The path to an XML file.

    Returns:
        df (Pandas DataFrame): XML data as a Pandas DataFrame.
    """
    
    def parse_xml(data, row_id = 'row'):
        """
        Parse XML data.
        """
        data_attribute = data.attrib
        for row in data.iter(row_id):
            row_dict = data_attribute.copy()
            row_dict.update(row.attrib)
            yield row_dict
            
    data = xml.parse(path)
    df = pd.DataFrame(list(parse_xml(data.getroot())))
    
    return df
