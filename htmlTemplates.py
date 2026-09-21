css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="http://localhost:8000/AI.png">
    </div>
    <div class="message"><span style="white-space: pre-line">{{MSG}}</span></div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="http://localhost:8000/users.png">
    </div>    
    <div class="message"><span style="white-space: pre-line">{{MSG}}</span></div>
</div>
'''

file_list_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="http://localhost:8000/users.png">
    </div>    
    <div class="message"><span style="white-space: pre-line">{{MSG}}</span></div>
</div>
'''

eli_bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="http://localhost:8000/icon.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

check_box = '''
   <div class="checkbox-container">
        <div id="custom-checkbox" style="width: 20px; height: 20px; border: 1px solid black; display: inline-block; cursor: pointer;" onclick="toggleCheckbox()"></div>
        <label for="custom-checkbox">Custom Div Checkbox</label>
   </div>
   <br>
    <input type="submit" value="Submit" onclick="return sendDivCheckboxValue();">
'''