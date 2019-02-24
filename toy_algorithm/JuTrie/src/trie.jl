"""
`jutrie` is a pure Julia implementation of the 
`trie` (prefix tree) data structure.
"""

import Base

mutable struct Node
    value
    # If the children is `Vector`
    # we can map it by Char's ASCII
    # Else if we want to support Unicode
    # we can Hierarchical it by `0` and `1`, deepen the Trie.
    # To simply,
    # we use the map structure
    children::Dict{Char, Node}

    Node(value=0) = new(value, Dict{Char, Node}())
end

struct __NULL end
NULL = __NULL()
isNULL(x) = (typeof(x) == __NULL)
# When we use the node, we need to check if it is NULL

struct Trie
    root::Node
    keyDict::Dict{String, Node}

    Trie() = new(Node(), Dict{String, Node}())
end

"""
    update!(trie::Trie, dataDict::Dict{String, Number})

Update the key and value by dataDict.
"""
function update!(trie::Trie, dataDict::Dict{String, T}) where {T <: Number}
    for (key, value) in dataDict
        set_value!(trie, key, value)
    end
    return trie
end

"""
    increase!(trie::Trie, key::String, value::Number) 

It will increase VALUE on key, 
if the key didn't exist,
we will initalize it by value.
"""
function increase!(trie::Trie, key::String, value::Number)
    set_value!(trie, key, exist_key(trie, key) ? get_value(trie, key)+value : value )
end

function set_value!(trie::Trie, key::String, value::Number)
    if ( !exist_key(trie, key) )
        node = trie.root
        for part in key
            if !(part in keys(node.children))
                node.children[part] = Node()
            end
            node = node.children[part]
        end
    else
        node = find_prefix_node(trie, key)
    end
    node.value = value
end

function get_value(trie::Trie, key::String)
    if ( !exist_key(trie, key) )
        error("KeyError: key $(key) not found")
    else
        return find_prefix_node(trie, key).value
    end
end

"""
    keys(trie::Trie) -> Array{String, 1}
Return the elements which not zero
"""
function Base.keys(trie::Trie)
    trieKeys = []

    function traversal(node, prefix::String)
        for (key, child) in node.children
            if child.value != 0
                append!(trieKeys, [string(prefix, key)])
            end
            traversal(child, string(prefix, key))
        end
    end

    traversal(trie.root, "")
    return trieKeys
end

function items(trie::Trie)

end

"""
    test_key(trie::Trie, key::String) -> Bool

The external interface to check if the key is valid.
"""
function test_key(trie::Trie, key::String)
    node = find_prefix_node(trie, key)
    return !isNULL(node) && (node.value > 0)
end

"""
    exist_key(trie::Trie, key::String) -> Bool

The internal interface to check if the key node is NULL.
"""
function exist_key(trie::Trie, key::String)
    return !isNULL( find_prefix_node(trie, key) )
end

function prefix_items(trie::Trie, prefix::String)
    node = find_prefix_node(trie, prefix)
    if isNULL(node) return [] end

    # Return all node's children element
    items = []
    function itemAppend(node, prefix)
        for (key, child) in node.children
            append!(items, (prefix+key, node.value))
            itemAppend(child)
        end
    end
    itemAppend(node)

    # Find the Real Keys
    return filter(p -> p[2] > 0, items)
end

function find_prefix_node(trie::Trie, prefix::String)
    # Find the prefix node
    node = trie.root
    for part in prefix
        if (part in keys(node.children))
            node = node.children[part]
        else return NULL end
    end
    return node
end

function longest_prefix_item(trie::Trie, prefix::String)
    longest_prefix_key, longest_prefix_value = NULL, NULL
    for (key, value) in prefix_items(trie, prefix)
        if (isNULL(longest_prefix_key)) || 
            (length(key) > length(longest_prefix_key))
            longest_prefix_value = value
            longest_prefix_key = key
        end
    end
    return ( isNULL(longest_prefix_key) ? () : 
            (longest_prefix_key, longest_prefix_value) )
end


function test_before()
    trie = Trie()
    update!(trie, Dict("abc" => 3, "all" => 1, "bad" => 4))
end

function test_keys()
    trie = test_before()
    show(keys(trie))
end

function test_get_value()
    trie = test_before()
    get_value(trie, "all") |> println
    try
        get_value(trie, "agc")
    catch
        println("agc not found")
    end
    increase!(trie, "all", 10) |> println
    get_value(trie, "all") |> println
end

function main()
    test_keys()
    test_get_value()
end

main()