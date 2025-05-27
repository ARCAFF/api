from sklearn.preprocessing import LabelEncoder

z_classes = ["A", "B", "C", "H", "LG"]
p_classes = ["asym", "r", "sym", "x"]
c_classes = ["frag", "o", "x"]

z_component_mapping = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "LG",  # Merge D, E, F into LG (LargeGroup)
    "E": "LG",
    "F": "LG",
    "H": "H",
}

p_component_mapping = {
    "x": "x",
    "r": "r",
    "s": "sym",  # Merge s and h into sym
    "h": "sym",
    "a": "asym",  # Merge a and k into asym
    "k": "asym",
}

c_component_mapping = {
    "x": "x",
    "o": "o",
    "i": "frag",  # Merge i and c into frag
    "c": "frag",
}


def create_encoders():
    """
    Create encoders from McIntosh component mappings.
    """
    # Create and fit encoders
    z_encoder = LabelEncoder()
    p_encoder = LabelEncoder()
    c_encoder = LabelEncoder()

    z_encoder.fit(z_classes)
    p_encoder.fit(p_classes)
    c_encoder.fit(c_classes)

    # Create encoders dictionary
    encoders = {"Z_encoder": z_encoder, "p_encoder": p_encoder, "c_encoder": c_encoder}

    # Create mappings dictionary
    mappings = {
        "Z_component": z_component_mapping,
        "p_component": p_component_mapping,
        "c_component": c_component_mapping,
    }

    return encoders, mappings
