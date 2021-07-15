function showVal(newVal){
document.getElementById("valBox").innerHTML=newVal;
}
function myChangeFunction(input1) {
var input2 = document.getElementById('myInput2');
input2.value = input1.value;
}
$('#dataset_type').change(function(){
   selection = $(this).val();
   switch(selection)
   {
       case 'Real':
           $('#dataset-real').show();
           $('#dataset-synthetic').hide();
           break;
       case 'Synthetic':
           $('#dataset-real').hide();
           $('#dataset-synthetic').show();
           break;
      default:
            $('#dataset-real').hide();
            $('#dataset-synthetic').hide();
            break;
   }
});
$(document).ready(function(){
  var multipleCancelButton = new Choices('#choices-multiple-remove-button', {
  removeItemButton: true,});
});
