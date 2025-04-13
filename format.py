from typing import Optional, Union
from dataclasses import dataclass
from typing_extensions import Protocol

class ResponseProtocol(Protocol):
    """Protocol defining the expected structure of a response object."""
    @property
    def content(self) -> str:
        
        ...

@dataclass
class FormattingOptions:
    """Configuration options for response formatting."""
    preserve_newlines: bool = True
    strip_metadata: bool = True
    code_block_style: str = "```"

def format_response_for_display(
    response_data: Optional[Union[ResponseProtocol, str, dict]],
    options: Optional[FormattingOptions] = None
) -> str:
    """
    Format the raw model response into clean markdown for display.
    Removes metadata and keeps only the actual content.
    
    Args:
        response_data: The raw response object from the model. Can be:
            - A response object with a content attribute
            - A string containing the raw content
            - A dictionary with a 'content' key
        options: Optional formatting configuration
        
    Returns:
        str: Formatted markdown response
        
    Raises:
        ValueError: If the response_data is invalid or cannot be processed
    """
    if options is None:
        options = FormattingOptions()

    try:
        # Extract content from different possible input types
        if response_data is None:
            raise ValueError("Response data cannot be None")
            
        if isinstance(response_data, str):
            content = response_data
        elif isinstance(response_data, dict):
            content = response_data.get('content', '')
            if not content:
                raise ValueError("No content found in response dictionary")
        elif hasattr(response_data, 'content'):
            content = response_data.content
        else:
            raise ValueError(f"Unsupported response type: {type(response_data)}")

        # Remove metadata if present
        if options.strip_metadata:
            # Remove the metadata section that starts with "additional_kwargs"
            if "additional_kwargs=" in content:
                content = content.split("additional_kwargs=")[0].strip()
            
            # Remove any remaining response metadata patterns
            metadata_patterns = [
                "response_metadata=",
                "id=run-",
                "usage_metadata=",
                "input_tokens:",
                "output_tokens:",
                "total_tokens:",
                "input_token_details:"
            ]
            
            for pattern in metadata_patterns:
                if pattern in content:
                    content = content.split(pattern)[0].strip()
            
            # Clean up any trailing special characters
            content = content.strip("{}[] \n\t")
            
            # Remove any metadata formatting
            content = content.replace("content='", "").replace("'", "")

        # Format code blocks and handle markdown
        lines = content.split('\\n' if '\\n' in content else '\n')
        formatted_lines = []
        in_code_block = False
        
        for line in lines:
            # Handle code block markers
            if options.code_block_style in line:
                in_code_block = not in_code_block
                # Preserve the language specification if present
                formatted_lines.append(line.replace("\\", ""))
            elif in_code_block:
                # Don't escape characters in code blocks
                formatted_lines.append(line.replace("\\", ""))
            else:
                # Process normal text lines
                processed_line = line.replace("\\", "")
                # Preserve markdown formatting
                processed_line = processed_line.replace("&lt;", "<").replace("&gt;", ">")
                formatted_lines.append(processed_line)
        
        # Join lines based on configuration
        if options.preserve_newlines:
            return "\n".join(formatted_lines)
        else:
            # Remove empty lines and join
            return "\n".join(line for line in formatted_lines if line.strip())

    except Exception as e:
        error_msg = f"Error formatting response: {str(e)}"
        return error_msg