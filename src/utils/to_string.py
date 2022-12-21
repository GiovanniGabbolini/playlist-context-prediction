"""
Created on Wed May 4 2022

@author Name Redacted Surname Redacted
"""


def to_string(d):
    """
    Convert a dictionary with floats as values to string.
    """
    assert all(type(v) == float for v in d.values())

    s = "{\n"
    for k, v in d.items():
        s += f"\t{k}: " + "{:.3f}".format(v) + "\n"
    s += "}"
    return s


if __name__ == "__main__":
    print(to_string({
        "a": 1.111111,
        "b": 2.222222,
    }))
