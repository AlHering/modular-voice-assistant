# Entity Data Interfaces
Configuration format for for handling data, data models, data backends and tools for data analysis and visualization.
The configuration format can be understood as as representation of a configuration format.
The format is also used for flexible and partly automated (re)-configuration for data handling.
Disclaimer: This functionality is used for personal projects and thus adjusted to subjectively often occuring tasks.


## Configuration Format
The profile of a configuration format is composed of 
- environment configuration
- entity configuration
- linkage configuration
- view configuration

It allows for handling data access with different logic and storage backend. The configuration format class functions as first layer interface.
Access operations are routed towards a second layer interface, declared within the environment configuration. 
The second layer interfaces operations to its specified storage system as third layer.

### Environment Configuration
The environment configuration is formatted as a JSON/dictionary profile.
An environment profile includes
- "backend", declaring a backend option ('database', 'filestore', 'plugin', ...).
- "framework", declaring a framework / typing ('sqlalchemy', 'json', ...) or the plugin name or path in case of plugin usage.
- "arguments", declaring framework arguments (database url, root file path, ...).
- "targets", declaring a list of entities, that the specific environment manages, or "*" as string to handle all entities

Example:
```json
{
  "backend": "database",
  "framework": "sqlalchemy",
  "arguments": {
    "database": "mysql://myUser:myPW@0.0.0.0:1234/my_schema",
    "dialect": "mysql",
    "encoding": "utf-8"
  },
  "targets": ["media", "position", "description", "file", "tag", "alias", "asset"]
}
```

### Entity Configuration
The entity configuration is formatted as a JSON/dictionary profile.
An entity profile includes
- a meta-data block (under '#meta') containing
  - "description": An entity-level description.
  - "schema": A schema for organizational entity separation.
  - "keep_deleted": Whether to keep deleted entries (in these cases you should add an attributed flag, with a default for delete calls to mark the soft-deleted entries)
  - "authorize", declaring an SHA-256 hash of the authorization password, only include if reading and writing needs to be authorized (use function `get_authorization_token` if you are unsure about the hashing)
  - "obfuscate", declaring a obfuscator plugin or an obfuscation lambda function as string
  - "deobfuscate", declaring a obfuscator plugin or an deobfuscation lambda function as string
- a key-value pair for each attribute containing a dictionary on the value side with
  - "type" (see Attribute Types for more information) (obligatory)
  - "description" containing a textual description (optional)
  - "key" declares, whether an attribute is a primary or part of a composite key (only needed if the key belongs to a primary or composite key)
  - "autoincrement" declares, whether an attribute should be autoincremented (only needed in case of autoincrement functionality)
  - "required" declares, whether attribute is not nullable (only needed if attribute is not nullable)
  - "post", "patch" and/or "delete", each containing a lambda function as string (getting the full entry data as single argument) for calculating a default value (only needed in case of the specific default value)
    (Note, for all lambda function strings are allowed to use Python's "datetime"-package.)

Note, that authorization, handled on Physcial Data Interface class, so the first layer while obfuscation and deobfuscation is handled on interface (second) layer.
Is a direct access to the second layer is given, the authorization can be bypassed.
Please be aware, what happens if using obfuscation without deobfuscation et vice versa: Either data is written obfuscated and retrieved in the same state.
Or data is written in clear form, but "deobfuscated" at retrieval. This behavior can be used for refactoring logic upon retrieval.
Please also be aware, that obfuscation methods are also used for transforming filter masks. 
So adjust your lambda functions or transformation plugins for different keys to be able to handle a large number of missing values for other keys.

Example:
```json
{
    "media": {
      "#meta": {
        "keep_deleted": true
      },
      "id": {
        "type": "int",
        "key": true,
        "autoincrement": true,
        "required": true,
        "description": "ID of the object."
      },
      "title": {
        "type": "text",
        "post": "lambda _: 'unkown'",
        "description": "Title of the media asset."
      },
      "type": {
        "type": "str",
        "post": "lambda _: 'media'",
        "description": "Type of the media asset: 'media', 'movie', 'game', 'book', ..."
      },
      "publishing_year": {
        "type": "int",
        "required": true,
        "description": "Publishing year of the media asset."
      },
      "score": {
        "type": "float_3_1",
        "description": "Score of the media asset, between 0.0 and 10.0."
      },
      "status": {
        "type": "str",
        "required": true,
        "description": "Status of the media asset: 'unknown' -> 'marked' -> 'acquired' -> 'seen' -> 'scored' -> 'favorited'",
        "post": "lambda _: 'unknown'"
      },
      "created": {
        "type": "datetime",
        "description": "Timestamp of creation.",
        "post": "lambda _: datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')"
      },
      "inactive": {
        "type": "char",
        "description": "Flag for marking inactive entries.",
        "delete": "lambda _: 'X'"
      }
    },
    "position": {
      "#meta": {
        "keep_deleted": true
      },
      "id": {
        "type": "int",
        "key": true,
        "autoincrement": true,
        "required": true,
        "description": "ID of the position."
      },
      "name": {
        "type": "text",
        "required": true,
        "description": "Name of the position."
      },
      "role": {
        "type": "text",
        "required": true,
        "description": "Role of the position."
      },
      "created": {
        "type": "datetime",
        "description": "Timestamp of creation.",
        "post": "lambda _: datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')"
      },
      "inactive": {
        "type": "char",
        "description": "Flag for marking inactive entries.",
        "delete": "lambda _: 'X'"
      }
    },
    "description": {
      "#meta": {
        "keep_deleted": true
      },
      "id": {
        "type": "int",
        "key": true,
        "autoincrement": true,
        "required": true,
        "description": "ID of the description."
      },
      "source": {
        "type": "text",
        "required": true,
        "description": "Source of the description."
      },
      "content": {
        "type": "longtext",
        "required": true,
        "description": "Content of the description."
      },
      "url": {
        "type": "text",
        "required": true,
        "description": "Source URL of the description."
      },
      "created": {
        "type": "datetime",
        "description": "Timestamp of creation.",
        "post": "lambda _: datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')"
      },
      "updated": {
        "type": "datetime",
        "description": "Timestamp of last update.",
        "post": "lambda _: datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')",
        "patch": "lambda _: datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')",
        "delete": "lambda _: datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')"
      },
      "inactive": {
        "type": "char",
        "description": "Flag for marking inactive entries.",
        "delete": "lambda _: 'X'"
      }
    },
    "file": {
      "#meta": {
        "keep_deleted": true
      },
      "id": {
        "type": "int",
        "key": true,
        "autoincrement": true,
        "required": true,
        "description": "ID of the file."
      },
      "file_name": {
        "type": "text",
        "required": true,
        "description": "Name of the file."
      },
      "file_type": {
        "type": "str",
        "required": true,
        "description": "Type of the file."
      },
      "path": {
        "type": "text",
        "required": true,
        "description": "Path of the file."
      },
      "origin": {
        "type": "str_120",
        "required": true,
        "description": "Origin of the file."
      },
      "quality": {
        "type": "str",
        "required": true,
        "description": "Quality of the file."
      },
      "url": {
        "type": "text",
        "description": "Source URL of the file."
      },
      "languages": {
        "type": "str_120",
        "required": true,
        "description": "Language of the file."
      },
      "runner": {
        "type": "text",
        "description": "Runner command for opening the file."
      },
      "created": {
        "type": "datetime",
        "description": "Timestamp of creation.",
        "post": "lambda _: datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')"
      },
      "updated": {
        "type": "datetime",
        "description": "Timestamp of last update.",
        "post": "lambda _: datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')",
        "patch": "lambda _: datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')",
        "delete": "lambda _: datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')"
      },
      "inactive": {
        "type": "char",
        "description": "Flag for marking inactive entries.",
        "delete": "lambda _: 'X'"
      }
    },
    "tag": {
      "id": {
        "type": "int",
        "key": true,
        "autoincrement": true,
        "required": true,
        "description": "ID of the tag."
      },
      "target_type": {
        "type": "str",
        "required": true,
        "description": "Target media type of the tag."
      },
      "name": {
        "type": "str",
        "required": true,
        "description": "Name of the tag."
      }
    },
    "alias": {
      "id": {
        "type": "int",
        "key": true,
        "autoincrement": true,
        "required": true,
        "description": "ID of the alias."
      },
      "name": {
        "type": "text",
        "required": true,
        "description": "Name of the alias."
      },
      "language": {
        "type": "str",
        "description": "Language of the alias."
      }
    },
    "asset": {
      "id": {
        "type": "int",
        "key": true,
        "autoincrement": true,
        "required": true,
        "description": "ID of the asset."
      },
      "binary": {
        "type": "blob",
        "required": true,
        "description": "Asset as binary large object."
      },
      "extension": {
        "type": "str",
        "required": true,
        "description": "Asset extension/format: 'png', 'jpeg', ..."
      }
    }
}
```



##### Linkage Configuration
The linkage configuration is formatted as a JSON/dictionary profile.
An linkage profile includes an nested block for each realtion under a given realtionship name. Inside this block,
the following information needs to be declared:
- "source": Source entity type
- "target": Target entity type
- "relation": "1:1", "1:n" or "n:m"
- "linkage_type": "foreign_key", "filter_mask" or "manual"
if "linkage_type" is "foreign_key" or "manual", the linkage ids must be declared as follows:
- "source_key", declaring a list of source key type and name
- "target_key", declaring a list of target key type and name
if "linkage_type" is "filter_masks", the linkage needs to be declared as follows:
- "linkage", declaring the filter masks, see Filter Masks for usage information

Example:
```json
{
    "features": {
      "source": "media",
      "target": "position",
      "relation": "1:n",
      "linkage_type": "foreign_key",
      "source_key": [
        "int",
        "id"
      ],
      "target_key": [
        "int",
        "id"
      ]
    },
    "tagged_with": {
      "source": "media",
      "target": "tag",
      "relation": "1:n",
      "linkage_type": "manual",
      "source_key": [
        "int",
        "id"
      ],
      "target_key": [
        "int",
        "id"
      ]
    },
    "described_with": {
      "source": "media",
      "target": "description",
      "relation": "1:n",
      "linkage_type": "foreign_key",
      "source_key": [
        "int",
        "id"
      ],
      "target_key": [
        "int",
        "id"
      ]
    },
    "acquired_as": {
      "source": "media",
      "target": "file",
      "relation": "1:n",
      "linkage_type": "foreign_key",
      "source_key": [
        "int",
        "id"
      ],
      "target_key": [
        "int",
        "id"
      ]
    },
    "also_named": {
      "source": "media",
      "target": "alias",
      "relation": "1:n",
      "linkage_type": "manual",
      "source_key": [
        "int",
        "id"
      ],
      "target_key": [
        "int",
        "id"
      ]
    },
  "cover": {
      "source": "media",
      "target": "asset",
      "relation": "1:1",
      "linkage_type": "manual",
      "source_key": [
        "int",
        "id"
      ],
      "target_key": [
        "int",
        "id"
      ]
    },
  "preview": {
      "source": "media",
      "target": "asset",
      "relation": "1:n",
      "linkage_type": "manual",
      "source_key": [
        "int",
        "id"
      ],
      "target_key": [
        "int",
        "id"
      ]
    }
}
```

##### View Configuration
The view configuration specifies in which structure the model is accessible. It consits of multiple view blocks, which contain
the structure and access definitions based on:
- "root": the root entity type
- "linkages": a list of linkages to heed

Furthermore there are optional, nested dictionaries for describing limitations under the linkage (or root entity type for the root entities) as key
- "filters": for filter masks 
- "authorization": for a SHA-256 hash which a prompted password is matched against
- "topics": an optional dictionary for organizing entities under topics, if the filter masks under a topic keyword are met

The following blocks include lambda functions. All functions get the entry data of each entry as input.
And last but not least the "representation" dictionary, containing
- "navigation": a lambda function, returning a textual representation of the entity
- "content": a nested dictionary, containing "table" and "cover" as nested dictionaries, containing
  - "validation": A lambda function for validating elements (if a second argument is supported by the function, the current active field path is forwarded as second argument)
  - "transformation": a lambda function, returning 
    - if the "type" is "table": a list, which is interpreted as table row
    - if the "type" is "cover": a list which is interpreted as cover image or a string, which is interpreted as cover image replacement 
  - "structure":
    - if the "type" is "table": a dictionary containing the table headers and whether their data should be intrepreted as label or a list of tags
    - if the "type" is "cover": a dictionary containing the resolution (as "[width]x[height]") and columns (as integer value)
- "info": a nested dictionary, using the key as field represenation title and containing
  - "field_path": the target data path as single key or list of keys for nested target values
  - "transformation": a transformation lambda function 
  - "type": visualization type, either 'label', 'tags', or 'images'

Note that the entity data includes linked entities as sub-fields. Therefore the entity data dictionary is a nested dictionary, potentially containing
a list of linked entity data dictionaries. 
Note that you do not have to implement both vcontent types (or even a single one). The types "Table" and "Cover" have defaults.
However, if you prefare one over another, either implement only this type or move it to the top of the content block.
Note that in the "info" block the first linked entity is used, if multiple are available, but no 'list(...)' is given as type.
Use that functionality in addition to filtering via "filters" to choose a single prefared entry, if necessary.

Example:
```json
{
  "MEDIA": {
    "root": "media",
    "linkages": ["tagged_with"],
    "filters": { 
      "media": [["inactive", "!=", "X"]],
      "tagged_with": [["inactive", "!=", "X"]]
    },
    "topics": {
      "movies": [["type", "==", "movie"]],
      "series": [["type", "==", "series"]],
      "games": [["type", "==", "game"]],
      "books": [["type", "==", "book"]],
      "documents": [["type", "==", "document"]],
      "lectures": [["type", "==", "lecture"]],
      "tutorial": [["type", "==", "tutorial"]],
      "videos": [["type", "==", "video"]]
    },
    "navigation": "lambda x: '[' + str(x['id']) + ']' + x['title']",
    "content": {
      "table": {
        "validation": "lambda x,y: 'id' in x and any(x['type'] in path for path in y)",
        "transformation": "lambda x: ['[' + str(x['id']) + ']' + x['title'], str(x['publishing_year']), str([y['name'] for y in x['tagged_with']])]",
        "structure": {
          "Name": "label",
          "Year": "label",
          "Tags": "tags"
        }
      },
      "cover": {
          "validation": "lambda x,y: 'id' in x and any(x['type'] in path for path in y)",
          "transformation": "lambda x: {'image_binary': x['cover'][0]['binary'], 'image_extension': x['cover'][0]['extension']} if x['cover'] else '[' + str(x['id']) + ']\\n' + x['title']",
          "structure": {"resolution": "100x300", "columns": 5 }
        }
    },
    "info": {
        "Title": {
          "field_path": "title",
          "type": "label"
        },
        "Year": {
          "field_path": "publishing_year",
          "transformation": "lambda x: str(x)",
          "type": "label"
        },
        "Cover": {
          "field_path": "cover",
          "transformation": "lambda x: [{'image_binary': elem['binary'], 'image_extension': elem['extension']} for elem in x]",
          "type": "images"
        },
        "Description": {
          "field_path": "described_with",
          "transformation": "lambda x: '[' + x[0]['source'] + ']\\n' + x[0]['content'] + '\\n(' + x[0]['url'] + ')' if x else ''",
          "type": "label"
        },
        "Tags": {
          "field_path": [
            "tagged_with",
            "name"
          ],
          "transformation": "lambda x: x",
          "type": "tags"
        },
        "Previews": {
          "field_path": "preview",
          "transformation": "lambda x: [{'image_binary': elem['binary'], 'image_extension': elem['extension']} for elem in x]",
          "type": "images"
        }
      }
    }
  }
```

### Further Information
#### Framework Arguments
The following sections explains the supported framework arguments for different frameworks:
- sqlalchemy
  - "database", declaring the database URI
  - "dialect", declaring the database dialect ("mysql", "sqlite", ...)
  - "encoding", declaring the encoding
- json
#### Attribute Types
Attribute Types are used to describe the data structure of an attribute. The options currently are
- "int": for an integer type
- "dict": for a json/dictionary type
- "datetime": for a datetime type
- "str": for a string of length 60
- "str_[X]": for a string of length [X]
- "text": for a text type
- "bool": for a boolean type
- "char": for a char type
- "longtext": for a longtext type
- "float": for a float of default length
- "float_[X]_[Y]": for a float of length [X] with [Y] digits after the decimal point

#### FilterMasks
FilterMasks are used for data constraining. A FilterMask contains a List of 

In cases of filter masks between multiple different entities (e.g. for linkage) the target value needs to be the target
attribute key (from the target entity), that should be used for comparison.

Comparison operators can be choosen from
```
COMPARISON_METHOD_DICTIONARY = {
    "equals": lambda x, y: x == y,
    "not_equals": lambda x, y: x != y,
    "contains": lambda x, y: y in x,
    "not_contains": lambda x, y: y not in x,
    "is_contained": lambda x, y: x in y,
    "not_is_contained": lambda x, y: x not in y,
    "==": lambda x, y: x == y,
    "!=": lambda x, y: x != y,
    "has": lambda x, y: y in x,
    "not_has": lambda x, y: y not in x,
    "in": lambda x, y: x in y,
    "not_in": lambda x, y: x not in y
}
```
