import ijson

def parse_input(fname):
    with open(fname) as f:
        for obj in ijson.items(f,"item"):
            user = obj["user"]
            entities_shortened = obj["entitiesShortened"]
            inputs = []
            for item in entities_shortened:
                if item["type"] == "userMention":
                    inputs.append("\1@"+item["value"]+"\1")
                elif item["type"] == "hashtag":
                    inputs.append("\2#"+item["value"]+"\2")
                elif item["type"] == "url":
                    inputs.append("\3<link>\3")
                else:
                    inputs.append(item["value"])
            entities_full = obj["entitiesFull"]
            expected_out = []
            for item in entities_full:
                if item["type"] == "url":
                    expected_out.append("%s")
                else:
                    expected_out.append(item["value"])

            yield "".join(inputs)," ".join(expected_out)
            

def input2training_batch(fname):
    for inputs, outputs in parse_input(fname):
        curr_buff = inputs+"\t"
        for c in outputs:
            print(curr_buff,c)
            if len(curr_buff) < 300:
                curr_buff = curr_buff + c
            else:
                curr_buff = inputs+"\t"
    
