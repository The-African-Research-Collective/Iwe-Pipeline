import sys
import unicodedata


def get_nonprintable_char_handler(replace_by: str = " ") -> str:
    """
    See https://huggingface.co/datasets/laurievb/OpenLID-v2/blob/main/scripts/tools/remove_non_printing_char.py
    """
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        # same as \p{C} in perl
        # see https://www.unicode.org/reports/tr44/#General_Category_Values
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line: str) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char
