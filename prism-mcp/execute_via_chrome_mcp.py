import subprocess
import json
import time

def setup_mcp(command):
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "python-client", "version": "1.0"}
        }
    }
    process.stdin.write(json.dumps(init_request) + "\n")
    process.stdin.flush()
    
    while True:
        line = process.stdout.readline()
        if not line: break
        resp = json.loads(line)
        if resp.get("id") == 1:
            initialized_notif = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            process.stdin.write(json.dumps(initialized_notif) + "\n")
            process.stdin.flush()
            break
    
    return process

def call_tool(process, tool_name, arguments={}):
    call_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }
    process.stdin.write(json.dumps(call_request) + "\n")
    process.stdin.flush()

    result = None
    while True:
        line = process.stdout.readline()
        if not line: break
        resp = json.loads(line)
        if resp.get("id") == 2:
            result = resp.get("result")
            break
    
    return result

if __name__ == "__main__":
    command = ["/usr/local/bin/npx", "-y", "chrome-devtools-mcp"]
    process = setup_mcp(command)
    
    js_code = """
(function fillShifts() {
  const dates = ['02/15/2026', '02/16/2026', '02/17/2026', '02/18/2026', '02/19/2026', '02/20/2026', '02/21/2026'];
  const saturdayDate = '02/21/2026';
  const overnightStart = '12:00 AM', overnightEnd = '03:00 AM', overnightCode = 'enhanced_enhanced_ps_overnight', locationValue = 'home';
  const wage1Start = '08:00 AM', wage1End = '06:00 PM', wageCode = 'enhanced_enhanced_ps_wage';
  const wage2Start = '08:00 AM', wage2End = '05:00 PM';
  
  const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
  const setInputValue = (input, value) => {
    if (!input) return;
    if (nativeInputValueSetter) { nativeInputValueSetter.call(input, value); } else { input.value = value; }
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
    input.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: 'Enter' }));
    input.dispatchEvent(new Event('blur', { bubbles: true }));
    if (typeof jQuery !== 'undefined') { jQuery(input).trigger('change'); jQuery(input).blur(); }
  };
  const setSelectValue = (select, value) => {
    if (!select) return;
    select.value = value;
    select.dispatchEvent(new Event('change', { bubbles: true }));
    if (typeof jQuery !== 'undefined' && jQuery(select).hasClass('select2-hidden-accessible')) { jQuery(select).trigger('change'); }
  };
  const fillSlot = (i, date, start, end, code, loc) => {
    const startInput = document.querySelector(`input[name="batch_shift_entry[batch_shift_entry_items_attributes][${i}][start_datetime]"]`);
    const endInput = document.querySelector(`input[name="batch_shift_entry[batch_shift_entry_items_attributes][${i}][end_datetime]"]`);
    const altStart = document.querySelector(`#start_datetime_input_${i} input[type="text"]`);
    const altEnd = document.querySelector(`#end_datetime_input_${i} input[type="text"]`);
    setInputValue(startInput, `${date} ${start}`); setInputValue(altStart, `${date} ${start}`);
    setInputValue(endInput, `${date} ${end}`); setInputValue(altEnd, `${date} ${end}`);
    const serviceSelect = document.querySelector(`select[name="batch_shift_entry[batch_shift_entry_items_attributes][${i}][service_code]"]`);
    setSelectValue(serviceSelect, code);
    const locSelect = document.querySelector(`select[name="batch_shift_entry[batch_shift_entry_items_attributes][${i}][additional_fields][service_location_internal_symbol]"]`);
    setSelectValue(locSelect, loc);
  };
  const getAddButton = () => Array.from(document.querySelectorAll('a, button')).find(el => el.textContent.trim().toLowerCase().includes('add another shift'));
  
  for (let i = 0; i < 7; i++) { fillSlot(i, dates[i], overnightStart, overnightEnd, overnightCode, locationValue); }
  let addButton = getAddButton();
  if (addButton) {
    addButton.click();
    setTimeout(() => {
      fillSlot(7, saturdayDate, wage1Start, wage1End, wageCode, locationValue);
      addButton = getAddButton();
      if (addButton) {
        addButton.click();
        setTimeout(() => { fillSlot(8, saturdayDate, wage2Start, wage2End, wageCode, locationValue); window.populateDone = true; }, 1000);
      }
    }, 1000);
  } else {
    fillSlot(7, saturdayDate, wage1Start, wage1End, wageCode, locationValue);
    try { fillSlot(8, saturdayDate, wage2Start, wage2End, wageCode, locationValue); } catch(e){}
    window.populateDone = true;
  }
  return "Script running";
})();
    """
    
    print("Evaluating JS through Chrome MCP...")
    res = call_tool(process, "evaluate_script", {"function": f"() => {{ {js_code} }}"})
    print(json.dumps(res, indent=2))
        
    process.terminate()
