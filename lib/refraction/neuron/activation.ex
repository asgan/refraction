defmodule Refraction.Neuron.Activation do
  @spec calculate_output(atom, input :: integer) :: float
  @spec calculate_output(atom, input :: [integer]) :: [float]
  def calculate_output(:identity, input), do: input
  def calculate_output(:relu, input), do: relu(input)
  def calculate_output(:sigmoid, input), do: sigmoid(input)
  def calculate_output(:softargmax, input), do: softargmax(input)
  def calculate_output(:tanh, input), do: tanh(input)

  defp relu(input) when input <= 0, do: 0
  defp relu(input) when input > 0, do: input

  defp sigmoid(input), do: 1 / (1 + :math.exp(-input))

  defp softargmax([input]), do: softargmax(input)

  defp softargmax(input) when is_list(input) and length(input) > 1 do
    maximum_entry_value = Enum.max(input)

    normalized_entry_values =
      Enum.map(input, fn entry_value -> entry_value - maximum_entry_value end)

    sum = sum_list(normalized_entry_values)

    Enum.map(normalized_entry_values, fn entry_value ->
      :math.exp(entry_value) / sum
    end)
  end

  defp softargmax(input), do: [input]

  defp tanh(input), do: :math.tanh(input)

  defp sum_list([]), do: 0
  defp sum_list([head | tail]), do: :math.exp(head) + sum_list(tail)
end
